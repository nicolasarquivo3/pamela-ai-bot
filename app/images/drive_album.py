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

def _shrink_image_bytes(data: bytes, max_side: int = 1280, quality: int = 82) -> bytes:
    """Reduz foto grande na memoria (celular 12MP+ derruba Render free)."""
    if not data or len(data) < 500:
        return data
    if len(data) < 350_000:
        return data
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
        w, h = im.size
        m = max(w, h)
        if m > max_side:
            scale = max_side / float(m)
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality, optimize=True)
        small = out.getvalue()
        if small and len(small) < len(data):
            return small
        return data
    except Exception as e:
        print(f"[DRIVE] shrink fail: {e}", flush=True)
        return data


def _tokenize(s: str) -> set:
    s = (s or "").lower()
    return {p for p in re.split(r"[\s\-_/,|]+", s) if len(p) > 2}


def _pedido_from_scene(scene: str) -> str:
    return scene or ""


def _outfit_from_scene(scene: str) -> str:
    return ""


try:
    from app.images.album_service import (
        _tokenize as _album_tok,
        _pedido_from_scene as _album_ped,
        _outfit_from_scene as _album_out,
    )
    _tokenize = _album_tok  # type: ignore
    _pedido_from_scene = _album_ped  # type: ignore
    _outfit_from_scene = _album_out  # type: ignore
except Exception:
    pass


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
        session_factory=None,  # SessionLocal — evita conflito async com outros loops
    ):
        self.session = session
        self.session_factory = session_factory
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

    def _own_session(self):
        """Sessao propria se session_factory existir (safe p/ background)."""
        if self.session_factory is not None:
            return self.session_factory()
        return None


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
        own = self._own_session()
        if own is not None:
            async with own as session:
                old = self.session
                self.session = session
                try:
                    await self.ensure_table()
                    r = await self.session.execute(
                        text("SELECT COUNT(*) FROM album_drive_photos")
                    )
                    return int(r.scalar() or 0)
                finally:
                    self.session = old
        await self.ensure_table()
        r = await self.session.execute(text("SELECT COUNT(*) FROM album_drive_photos"))
        return int(r.scalar() or 0)

    async def stats(self) -> dict:
        """total / com tag / sem tag."""
        await self.ensure_table()
        own = self._own_session() if hasattr(self, "_own_session") else None

        async def _run(session):
            r = await session.execute(
                text(
                    """
                    SELECT
                      COUNT(*)::int AS total,
                      COUNT(*) FILTER (
                        WHERE caption IS NOT NULL AND TRIM(caption) <> ''
                      )::int AS tagged,
                      COUNT(*) FILTER (
                        WHERE caption IS NULL OR TRIM(COALESCE(caption,'')) = ''
                      )::int AS untagged
                    FROM album_drive_photos
                    """
                )
            )
            row = r.mappings().first() or {}
            total = int(row.get("total") or 0)
            tagged = int(row.get("tagged") or 0)
            untagged = int(row.get("untagged") or 0)
            return {
                "total": total,
                "tagged": tagged,
                "untagged": untagged,
                "pct": round(100.0 * tagged / total, 1) if total else 0.0,
            }

        if own is not None:
            async with own as session:
                old = self.session
                self.session = session
                try:
                    return await _run(session)
                finally:
                    self.session = old
        return await _run(self.session)


    async def sync(self, limit: int = 200, caption_new: bool = True) -> dict:
        """Lista pasta Drive e indexa ate `limit` arquivos novos/alterados."""
        if not self.enabled:
            return {"ok": False, "error": "drive_disabled"}
        # sessao dedicada no background (nao reusa a do request)
        own = self._own_session()
        if own is not None:
            async with own as session:
                old = self.session
                self.session = session
                try:
                    return await self._sync_body(limit, caption_new)
                finally:
                    self.session = old
        return await self._sync_body(limit, caption_new)


    async def download_bytes(self, file_id: str) -> bytes | None:
        """Baixa bytes de um arquivo do Drive (service account)."""
        try:
            import io
            from googleapiclient.http import MediaIoBaseDownload

            svc = self._build_service()
            req = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            data = buf.getvalue()
            if data and len(data) > 100:
                # sempre encolhe pra caption/swap nao estourar RAM
                return _shrink_image_bytes(data, max_side=1280)
            return None
        except Exception as e:
            print(f"[DRIVE] download fail {file_id[:12]}: {e}", flush=True)
            return None

    async def _sync_body(self, limit: int = 200, caption_new: bool = True) -> dict:
        """
        Indexa ate `limit` fotos NOVAS (ainda nao no banco).
        Fotos ja indexadas sao puladas e NAO consomem a cota do limit.
        Assim pastas grandes (milhares) vao enchendo a cada /album_drive_sync.
        """
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
        skipped = 0
        page_token = None
        listed = 0
        # safety: max pages to avoid infinite
        max_pages = 50
        pages = 0

        while added < limit and pages < max_pages:
            pages += 1
            page_size = 100
            try:
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
            except Exception as e:
                print(f"[DRIVE] list fail: {e}", flush=True)
                return {
                    "ok": False,
                    "error": f"list:{e}",
                    "added": added,
                    "updated": updated,
                    "captioned": captioned,
                    "skipped": skipped,
                    "listed": listed,
                }

            files = resp.get("files") or []
            if not files:
                break

            for f in files:
                listed += 1
                fid = f["id"]
                name = f.get("name") or ""
                mime = f.get("mimeType") or "image/jpeg"
                size = int(f.get("size") or 0)

                r = await self.session.execute(
                    text(
                        "SELECT id, caption FROM album_drive_photos "
                        "WHERE drive_file_id=:fid"
                    ),
                    {"fid": fid},
                )
                row = r.mappings().first()

                if row is not None:
                    # ja indexada: so toca se faltar caption e caption_new
                    if (
                        caption_new
                        and self.use_vision_caption
                        and self._caption_fn
                        and not (row.get("caption") or "").strip()
                        and added < limit
                    ):
                        caption = ""
                        try:
                            raw = await self.download_bytes(fid)
                            if raw:
                                caption = await self._caption_fn(raw) or ""
                                if caption:
                                    captioned += 1
                                    await self.session.execute(
                                        text(
                                            "UPDATE album_drive_photos SET caption=:c, "
                                            "tags=:t, synced_at=NOW() "
                                            "WHERE drive_file_id=:fid"
                                        ),
                                        {
                                            "c": caption[:2000],
                                            "t": " ".join(
                                                sorted(_tokenize(f"{name} {caption}"))[
                                                    :32
                                                ]
                                            ),
                                            "fid": fid,
                                        },
                                    )
                                    added += 1  # conta como trabalho do batch
                                    print(
                                        f"[DRIVE] backfill cap {name[:40]!r} -> {caption[:50]!r}",
                                        flush=True,
                                    )
                        except Exception as e:
                            print(f"[DRIVE] caption fail: {e}", flush=True)
                    else:
                        skipped += 1
                    continue

                # NOVA — indexar
                if added >= limit:
                    break

                caption = ""
                tags = " ".join(sorted(_tokenize(name))[:32])
                if caption_new and self.use_vision_caption and self._caption_fn:
                    try:
                        raw = await self.download_bytes(fid)
                        if raw and self._caption_fn:
                            caption = await self._caption_fn(raw) or ""
                            if caption:
                                captioned += 1
                                tags = " ".join(
                                    sorted(_tokenize(f"{name} {caption}"))[:32]
                                )
                                print(
                                    f"[DRIVE] caption {name[:40]!r} -> {caption[:50]!r}",
                                    flush=True,
                                )
                    except Exception as e:
                        print(f"[DRIVE] caption fail: {e}", flush=True)

                await self.session.execute(
                    text(
                        """
                        INSERT INTO album_drive_photos
                        (drive_file_id, name, mime_type, size_bytes, caption, tags, synced_at)
                        VALUES (:fid, :n, :m, :s, :c, :t, NOW())
                        ON CONFLICT (drive_file_id) DO UPDATE SET
                          name=EXCLUDED.name,
                          mime_type=EXCLUDED.mime_type,
                          size_bytes=EXCLUDED.size_bytes,
                          caption=CASE
                            WHEN EXCLUDED.caption <> '' THEN EXCLUDED.caption
                            ELSE album_drive_photos.caption
                          END,
                          tags=CASE
                            WHEN EXCLUDED.tags <> '' THEN EXCLUDED.tags
                            ELSE album_drive_photos.tags
                          END,
                          synced_at=NOW()
                        """
                    ),
                    {
                        "fid": fid,
                        "n": name[:500],
                        "m": mime[:120],
                        "s": size,
                        "c": (caption or "")[:2000],
                        "t": tags[:1000],
                    },
                )
                added += 1
                print(f"[DRIVE] +nova {added}/{limit} {name[:50]!r}", flush=True)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break
            if added >= limit:
                break

        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()

        total = await self.count()
        out = {
            "ok": True,
            "scanned": listed,
            "listed": listed,
            "added": added,
            "updated": updated,
            "captioned": captioned,
            "skipped": skipped,
            "total": total,
            "pages": pages,
        }
        print(f"[DRIVE] sync done {out}", flush=True)
        return out


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
        own = self._own_session()
        if own is not None:
            async with own as session:
                old = self.session
                self.session = session
                try:
                    return await self._pick_best_body(scene)
                finally:
                    self.session = old
        return await self._pick_best_body(scene)

    async def _pick_best_body(self, scene: str) -> dict | None:
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
        """Gera caption IA so para fotos que ainda nao tem legenda."""
        if not self._caption_fn:
            print("[DRIVE] backfill: sem caption_fn", flush=True)
            return 0
        own = self._own_session() if hasattr(self, "_own_session") else None
        if own is not None:
            async with own as session:
                old = self.session
                self.session = session
                try:
                    return await self._backfill_body(limit)
                finally:
                    self.session = old
        return await self._backfill_body(limit)

    async def _backfill_body(self, limit: int = 20) -> int:
        await self.ensure_table()
        r = await self.session.execute(
            text(
                """
                SELECT id, drive_file_id, name FROM album_drive_photos
                WHERE caption IS NULL OR TRIM(COALESCE(caption, '')) = ''
                ORDER BY id ASC
                LIMIT :lim
                """
            ),
            {"lim": max(1, min(int(limit), 20))},
        )
        rows = list(r.fetchall())
        done = 0
        for row in rows:
            pid, fid, name = row[0], row[1], (row[2] or "")
            try:
                raw = None
                if hasattr(self, "download_bytes"):
                    raw = await self.download_bytes(fid)
                if not raw and hasattr(self, "download"):
                    raw = await self.download(fid)
                if not raw:
                    print(f"[DRIVE] backfill skip no bytes {name[:40]!r}", flush=True)
                    continue
                cap = await self._caption_fn(raw) or ""
                try:
                    del raw
                except Exception:
                    pass
                if not cap.strip():
                    continue
                tags = " ".join(sorted(_tokenize(f"{name} {cap}"))[:32])
                await self.session.execute(
                    text(
                        "UPDATE album_drive_photos SET caption=:c, tags=:t, "
                        "synced_at=NOW() WHERE id=:id"
                    ),
                    {"c": cap[:2000], "t": tags[:1000], "id": pid},
                )
                done += 1
                print(
                    f"[DRIVE] backfill {done}/{limit} {name[:40]!r} -> {cap[:50]!r}",
                    flush=True,
                )
            except Exception as e:
                print(f"[DRIVE] backfill fail id={pid}: {e}", flush=True)
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
        return done

