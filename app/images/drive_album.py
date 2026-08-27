"""
Album no Google Drive (gratis).

Fluxo:
  1) Voce sobe as fotos numa pasta do Drive (PC ou backup do celular)
  2) Compartilha a pasta com o e-mail da service account
  3) Bot: /album_drive_sync  -> indexa + caption IA
  4) /foto usa Drive (baixa -> face swap -> envia)
  5) Pode apagar as fotos do celular — ficam so no Drive

Env:
  GOOGLE_DRIVE_FOLDER_ID=...
  GOOGLE_SERVICE_ACCOUNT_JSON={"type":"service_account",...}   # JSON inteiro numa linha
  # ou GOOGLE_SERVICE_ACCOUNT_FILE=/path (se montar secret)
  DRIVE_ALBUM_ENABLED=true
  DRIVE_ALBUM_FIRST=true   # Drive antes do canal Telegram
"""
from __future__ import annotations

import io
import json
import os
import random
import re
from typing import Any

from sqlalchemy import text

# reusa tokenize/sinonimos do album se disponivel
try:
    from app.images.album_service import _tokenize, _pedido_from_scene, _outfit_from_scene
except Exception:
    def _tokenize(s: str) -> set:
        s = (s or "").lower()
        return {p for p in re.split(r"[\s\-_/,|]+", s) if len(p) > 2}

    def _pedido_from_scene(scene: str) -> str:
        return scene or ""

    def _outfit_from_scene(scene: str) -> str:
        return ""


class DriveAlbumService:
    def __init__(
        self,
        session,
        folder_id: str | None = None,
        sa_json: str | None = None,
        enabled: bool = True,
        use_vision_caption: bool = True,
        caption_fn=None,  # async (bytes) -> str  (do AlbumService)
        download_caption_from_album=None,
    ):
        self.session = session
        self.folder_id = (folder_id or "").strip() or None
        self.sa_json = (sa_json or "").strip() or None
        self.enabled = bool(enabled) and bool(self.folder_id) and bool(self.sa_json)
        self.use_vision_caption = bool(use_vision_caption)
        self._caption_fn = caption_fn
        self._ready = False
        self._service = None
        self._recent: list[str] = []

    async def available(self) -> bool:
        return self.enabled

    def _build_service(self):
        if self._service is not None:
            return self._service
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        info = json.loads(self.sa_json)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    async def ensure_table(self) -> None:
        if self._ready:
            return
        await self.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS album_drive_photos (
                    id BIGSERIAL PRIMARY KEY,
                    drive_file_id TEXT NOT NULL UNIQUE,
                    name TEXT DEFAULT '',
                    mime_type TEXT DEFAULT '',
                    caption TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    size_bytes BIGINT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    synced_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        )
        await self.session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_album_drive_synced "
                "ON album_drive_photos (synced_at DESC)"
            )
        )
        await self.session.commit()
        self._ready = True
        print("[DRIVE] tabela album_drive_photos OK", flush=True)

    async def count(self) -> int:
        if not self.enabled:
            return 0
        await self.ensure_table()
        r = await self.session.execute(text("SELECT COUNT(*) FROM album_drive_photos"))
        return int(r.scalar() or 0)

    async def sync(self, limit: int = 200, caption_new: bool = True) -> dict:
        """Lista pasta Drive e indexa ate `limit` arquivos novos/alterados."""
        if not self.enabled:
            return {"ok": False, "error": "drive_disabled"}
        await self.ensure_table()
        try:
            svc = self._build_service()
        except Exception as e:
            print(f"[DRIVE] auth fail: {e}", flush=True)
            return {"ok": False, "error": f"auth:{e}"}

        q = (
            f"'{self.folder_id}' in parents and trashed=false and "
            f"(mimeType contains 'image/')"
        )
        added = 0
        updated = 0
        captioned = 0
        page_token = None
        scanned = 0

        while scanned < limit:
            page_size = min(100, limit - scanned)
            resp = (
                svc.files()
                .list(
                    q=q,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageSize=page_size,
                    pageToken=page_token,
                    orderBy="modifiedTime desc",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files = resp.get("files") or []
            if not files:
                break
            for f in files:
                scanned += 1
                fid = f["id"]
                name = f.get("name") or ""
                mime = f.get("mimeType") or "image/jpeg"
                size = int(f.get("size") or 0)
                # ja existe?
                r = await self.session.execute(
                    text(
                        "SELECT id, caption FROM album_drive_photos "
                        "WHERE drive_file_id=:fid"
                    ),
                    {"fid": fid},
                )
                row = r.mappings().first()
                caption = ""
                tags = ""
                if row and (row.get("caption") or "").strip():
                    # so atualiza synced
                    await self.session.execute(
                        text(
                            "UPDATE album_drive_photos SET synced_at=NOW(), "
                            "name=:n, mime_type=:m, size_bytes=:s "
                            "WHERE drive_file_id=:fid"
                        ),
                        {"n": name, "m": mime, "s": size, "fid": fid},
                    )
                    updated += 1
                else:
                    if caption_new and self.use_vision_caption and self._caption_fn:
                        raw = self._download_bytes(svc, fid)
                        if raw and self._caption_fn:
                            try:
                                caption = await self._caption_fn(raw) or ""
                                if caption:
                                    captioned += 1
                                    print(
                                        f"[DRIVE] caption {name[:40]!r} -> {caption[:50]!r}",
                                        flush=True,
                                    )
                            except Exception as e:
                                print(f"[DRIVE] caption fail: {e}", flush=True)
                    tags = " ".join(sorted(_tokenize(f"{name} {caption}"))[:32])
                    await self.session.execute(
                        text(
                            """
                            INSERT INTO album_drive_photos
                              (drive_file_id, name, mime_type, caption, tags, size_bytes)
                            VALUES
                              (:fid, :n, :m, :c, :t, :s)
                            ON CONFLICT (drive_file_id) DO UPDATE SET
                              name=EXCLUDED.name,
                              caption=CASE
                                WHEN EXCLUDED.caption<>'' THEN EXCLUDED.caption
                                ELSE album_drive_photos.caption
                              END,
                              tags=CASE
                                WHEN EXCLUDED.tags<>'' THEN EXCLUDED.tags
                                ELSE album_drive_photos.tags
                              END,
                              synced_at=NOW()
                            """
                        ),
                        {
                            "fid": fid,
                            "n": name,
                            "m": mime,
                            "c": caption,
                            "t": tags,
                            "s": size,
                        },
                    )
                    added += 1
                await self.session.commit()
                if scanned >= limit:
                    break
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        print(
            f"[DRIVE] sync scanned={scanned} added={added} "
            f"updated={updated} captioned={captioned}",
            flush=True,
        )
        return {
            "ok": True,
            "scanned": scanned,
            "added": added,
            "updated": updated,
            "captioned": captioned,
            "total": await self.count(),
        }

    def _download_bytes(self, svc, file_id: str) -> bytes | None:
        try:
            from googleapiclient.http import MediaIoBaseDownload

            buf = io.BytesIO()
            req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
            downloader = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            data = buf.getvalue()
            if len(data) < 500:
                return None
            return data
        except Exception as e:
            print(f"[DRIVE] download {file_id[:12]}: {e}", flush=True)
            return None

    async def download(self, drive_file_id: str) -> bytes | None:
        try:
            svc = self._build_service()
            return self._download_bytes(svc, drive_file_id)
        except Exception as e:
            print(f"[DRIVE] download err: {e}", flush=True)
            return None

    def _score(self, row, pedido_toks, outfit_toks) -> float:
        blob = f"{row.get('caption') or ''} {row.get('tags') or ''} {row.get('name') or ''}".lower()
        photo_toks = _tokenize(blob)
        score = 0.0
        if pedido_toks:
            score += len(pedido_toks & photo_toks) * 3.0
            for t in pedido_toks:
                if t in blob:
                    score += 1.5
        if outfit_toks:
            score += len(outfit_toks & photo_toks) * 1.0
        if row.get("drive_file_id") in self._recent:
            score -= 2.5
        if not photo_toks:
            score = 0.05
        return score

    async def pick_best(self, scene: str) -> dict | None:
        if not self.enabled:
            return None
        await self.ensure_table()
        r = await self.session.execute(
            text(
                "SELECT id, drive_file_id, name, caption, tags "
                "FROM album_drive_photos ORDER BY id DESC LIMIT 2000"
            )
        )
        rows = [dict(x) for x in r.mappings().all()]
        if not rows:
            print("[DRIVE] vazio — rode /album_drive_sync", flush=True)
            return None

        pedido = _pedido_from_scene(scene)
        outfit = _outfit_from_scene(scene)
        pedido_toks = _tokenize(pedido)
        outfit_toks = _tokenize(outfit)
        print(
            f"[DRIVE] pick pedido={pedido[:60]!r} toks={list(pedido_toks)[:10]}",
            flush=True,
        )

        scored = [(self._score(row, pedido_toks, outfit_toks), row) for row in rows]
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][0]
        print(
            f"[DRIVE] candidatos={len(rows)} top={[round(s,2) for s,_ in scored[:5]]}",
            flush=True,
        )

        if best < 0.5:
            pool = [
                d for d in rows[:40] if d.get("drive_file_id") not in self._recent
            ] or rows[:40]
            chosen = random.choice(pool)
            print("[DRIVE] score baixo -> aleatoria", flush=True)
        else:
            pool = [d for s, d in scored[:8] if s >= best - 1.0] or [scored[0][1]]
            chosen = random.choice(pool)

        fid = chosen["drive_file_id"]
        raw = await self.download(fid)
        if not raw:
            print("[DRIVE] download falhou", flush=True)
            return None

        self._recent.append(fid)
        self._recent = self._recent[-15:]
        print(
            f"[DRIVE] escolhida id={chosen.get('id')} name={chosen.get('name')!r} "
            f"cap={str(chosen.get('caption') or '')[:40]!r} bytes={len(raw)}",
            flush=True,
        )
        return {
            "drive_file_id": fid,
            "image_bytes": raw,
            "caption": chosen.get("caption") or "",
            "name": chosen.get("name") or "",
            "id": chosen.get("id"),
        }

    async def backfill_captions(self, limit: int = 20) -> int:
        await self.ensure_table()
        if not self._caption_fn:
            return 0
        r = await self.session.execute(
            text(
                """
                SELECT id, drive_file_id FROM album_drive_photos
                WHERE caption IS NULL OR caption = ''
                ORDER BY id DESC LIMIT :lim
                """
            ),
            {"lim": int(limit)},
        )
        rows = list(r.mappings().all())
        done = 0
        for row in rows:
            raw = await self.download(row["drive_file_id"])
            if not raw:
                continue
            cap = await self._caption_fn(raw)
            if not cap:
                continue
            tags = " ".join(sorted(_tokenize(cap))[:32])
            await self.session.execute(
                text(
                    "UPDATE album_drive_photos SET caption=:c, tags=:t WHERE id=:id"
                ),
                {"c": cap, "t": tags, "id": row["id"]},
            )
            await self.session.commit()
            done += 1
            print(f"[DRIVE] backfill id={row['id']} cap={cap[:50]!r}", flush=True)
        return done
