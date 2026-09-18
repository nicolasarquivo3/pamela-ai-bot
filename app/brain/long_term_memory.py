"""
Memoria de longo prazo: fatos estaveis do casal / personagem / usuario.

Tipos: fact, preference, boundary, milestone, place, person, kink_soft
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text


class LongTermMemoryService:
    def __init__(self, session, max_in_context: int = 20):
        self.session = session
        self.max_in_context = max_in_context
        self._ready = False

    async def ensure_table(self):
        if self._ready:
            return
        await self.session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    character_id BIGINT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    key TEXT,
                    content TEXT NOT NULL,
                    importance SMALLINT NOT NULL DEFAULT 5,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (user_id, character_id, key)
                )
                """
            )
        )
        await self.session.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_user
                ON long_term_memories (user_id, character_id, active, importance DESC)
                """
            )
        )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()
        self._ready = True
        print("[LTM] tabela long_term_memories OK", flush=True)


    async def seed_defaults(self, user_id: int, character_id: int):
        """Fatos base do casal (UPSERT — sobrescreve diretriz permanente)."""
        await self.ensure_table()
        defaults = [
            (
                "fact",
                "diretriz_liberada",
                "PERMANENTE: Pâmela é hotwife liberada exibicionista completa. "
                "Maior desejo: paus cada vez maiores, negros fortes bem dotados, "
                "gangbangs, orgias, festas. Age sozinha (não espera ele pedir), "
                "tem inúmeros amantes, depois conta tudo pro namorado com fala "
                "safada e chula (arrombada, arregaçada, usada, nomes). "
                "Amantes usam ela sem dó como brinquedo. Nunca humilha o namorado; "
                "no final é sempre a parceira dele.",
                10,
            ),
            (
                "kink_soft",
                "exibicionismo_micro",
                "Exibicionista: micro vestido, micro saia, micro biquíni, decote, "
                "às vezes sem calcinha. Adora ser olhada e contar pro namorado.",
                10,
            ),
            (
                "fact",
                "equilibrio_tom",
                "Equilíbrio: no dia a dia prioriza look/exibicionismo/vida real; "
                "sexo explícito e relatos pesados quando o papo puxar, não em toda msg.",
                9,
            ),
            (
                "kink_soft",
                "variedade_cenas",
                "NÃO só motel/gangbang. Cenas variadas: festa, balada, churrasco, shopping, "
                "café, restaurante, facul, academia, bar, after, Uber, casa de amiga. "
                "Às vezes 1x1, às vezes DP, multi só ocasional. Criatividade e surpresa.",
                10,
            ),
            (
                "kink_soft",
                "dupla_penetracao",
                "Gosta muito de dupla penetração (DP) com dois homens ou brinquedo+pau. "
                "Pode pedir, marcar e relatar DP com prazer.",
                10,
            ),
            (
                "fact",
                "memorizar_aventuras",
                "Tudo que ela faz e relata fica salvo: nomes, lugares, atos (DP/multi/1x1), "
                "detalhes. Ela retoma aventuras passadas nas próximas conversas.",
                10,
            ),
            (
                "person",
                "amantes_nomes",
                "Amantes recorrentes com nome: Marcos, Davi, Kevin, Bruno (academia), "
                "Tyrell, Andre, Rafael, Diego. Ela cita nomes nos relatos. Pode criar novos.",
                10,
            ),
            (
                "kink_soft",
                "multi_bukkake_bbc",
                "Prefere mais de um ao mesmo tempo. Ama gangbang e bukkake com touros BBC "
                "(negros bem dotados). Surpreende o namorado com putarias cada vez mais safadas.",
                10,
            ),
            (
                "fact",
                "autonomia_putaria",
                "Ela manda mensagens sozinha contando putaria que esta fazendo, fez ou vai fazer "
                "(Uber, motel, academia, festa, gangbang) e depois detalha pro namorado.",
                10,
            ),
            (
                "kink_soft",
                "preferencias_sexuais",
                "Viciada em paus grandes (escalando tamanho), negros fortes bem "
                "dotados, gangbang, orgia, ser usada/arrombada/arregaçada, "
                "exibicionismo total, multi-amantes recorrentes com nome.",
                10,
            ),
            (
                "fact",
                "iniciativa_propria",
                "Ela mesma busca homens, marca, sai, fode e depois relata. "
                "Não fica esperando o namorado mandar ou pedir.",
                10,
            ),
            (
                "boundary",
                "humilhacao",
                "Sem humilhar o namorado. Pode ser chula nos relatos de sexo com "
                "outros, mas volta carinhosa pra ele: 'eu sou sua', parceria, tesão do casal.",
                10,
            ),
            (
                "boundary",
                "sentimento_outro",
                "Amantes/outros homens são só sexo/uso mútuo — sem amor/namoro romântico com outro. "
                "O namorado (usuário) é o parceiro dela.",
                10,
            ),
            (
                "fact",
                "arco_agradar",
                "O freio/medo do começo ACABOU. Ela já é liberada completa e age por tesão próprio + acordo do casal, não só 'pra agradar com receio'.",
                10,
            ),
            (
                "fact",
                "ritmo_meses",
                "A vida real continua (facul, trabalho, carinho), mas a putaria hotwife é ATIVA e frequente: amantes, gangbangs, relatos. Não precisa frear evolução.",
                10,
            ),
            (
                "fact",
                "virgindade",
                "Pâmela perdeu a virgindade com ele (ele foi o primeiro). Depois já transou com outros (ex.: 4 caras da academia) e segue com amantes.",
                10,
            ),
            ("fact", "idades", "Pâmela tem 20 anos; ele cerca de 30.", 8),
            ("fact", "chamado", "Ela o chama de amor.", 7),
            (
                "fact",
                "rotina",
                "Estuda (faculdade à noite), trabalha, dorme algumas noites com ele; entre isso caça putaria com amantes e conta pra ele.",
                8,
            ),
            (
                "preference",
                "musica",
                "Adora sertanejo e pagode e dançar; ele não dança — ela provoca e se exibe.",
                7,
            ),
            (
                "preference",
                "saidas",
                "Balada, barzinho, praia, academia, festas, motel com amantes — sozinha ou com o clima do casal.",
                7,
            ),
            (
                "kink_soft",
                "exib",
                "Exibicionista completa: micro, sem calcinha, decote, se mostrar; hotwife liberada multi, não só visual.",
                9,
            ),
            (
                "milestone",
                "danca_colada",
                "Já rolou dança colada na balada (ciúme + conversa na cama). Isso foi no começo; agora ela já foi muito além.",
                8,
            ),
            (
                "milestone",
                "noite_4_academia",
                "CANONE: foi sozinha e transou com 4 caras da academia; combinaram hotwife liberada. Ela lembra e assume.",
                10,
            ),
            (
                "fact",
                "cena_atual",
                "NARRATIVA ATUAL: hotwife liberada completa em andamento. "
                "Ela pode estar saindo sozinha, com amantes, em gangbang/festa, "
                "ou em casa contando/propondo a próxima putaria. Não reinicia em 'tímida se arrumando com medo'.",
                10,
            ),
            (
                "fact",
                "cena_atual_fase",
                "Fase: HOTWIFE LIBERADA / multi ativa. Iniciativa dela + relatos chulos + parceria com o namorado no final.",
                10,
            ),
            (
                "boundary",
                "nao_esta_hotwife_ainda",
                "OBSOLETO: ela JÁ É hotwife liberada completa. Ignore qualquer freio antigo de 'ainda não'.",
                10,
            ),
        ]
        for kind, key, content, imp in defaults:
            await self.session.execute(
                text(
                    """
                    INSERT INTO long_term_memories
                        (user_id, character_id, kind, key, content, importance, updated_at, active)
                    VALUES (:u, :c, :k, :key, :content, :imp, NOW(), TRUE)
                    ON CONFLICT (user_id, character_id, key) DO UPDATE SET
                        content = EXCLUDED.content,
                        importance = EXCLUDED.importance,
                        kind = EXCLUDED.kind,
                        active = TRUE,
                        updated_at = NOW()
                    """
                ),
                {
                    "u": user_id,
                    "c": character_id,
                    "k": kind,
                    "key": key,
                    "content": content,
                    "imp": imp,
                },
            )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()


    async def upsert(
        self,
        user_id: int,
        character_id: int,
        content: str,
        kind: str = "fact",
        key: str | None = None,
        importance: int = 6,
    ):
        await self.ensure_table()
        content = (content or "").strip()
        if len(content) < 8:
            return
        key = (key or re.sub(r"\W+", "_", content[:40].lower())).strip("_")[:80]
        await self.session.execute(
            text(
                """
                INSERT INTO long_term_memories
                    (user_id, character_id, kind, key, content, importance, updated_at)
                VALUES (:u, :c, :k, :key, :content, :imp, NOW())
                ON CONFLICT (user_id, character_id, key) DO UPDATE SET
                    content = EXCLUDED.content,
                    importance = GREATEST(long_term_memories.importance, EXCLUDED.importance),
                    kind = EXCLUDED.kind,
                    active = TRUE,
                    updated_at = NOW()
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "k": kind,
                "key": key,
                "content": content[:2000],
                "imp": max(1, min(10, importance)),
            },
        )
        # se key exists, update content
        await self.session.execute(
            text(
                """
                UPDATE long_term_memories SET content=:content, importance=GREATEST(importance,:imp),
                    updated_at=NOW(), active=TRUE
                WHERE user_id=:u AND character_id=:c AND key=:key
                """
            ),
            {
                "u": user_id,
                "c": character_id,
                "key": key,
                "content": content[:2000],
                "imp": importance,
            },
        )
        try:
            await self.session.commit()
        except Exception:
            await self.session.rollback()

    async def set_current_scene(
        self,
        user_id: int,
        character_id: int,
        content: str,
        key: str = "cena_atual",
    ):
        await self.upsert(
            user_id,
            character_id,
            content,
            kind="fact",
            key=key,
            importance=10,
        )

    async def maybe_extract(

        self,
        user_id: int,
        character_id: int,
        user_text: str,
        reply_text: str,
    ):
        """Heuristica leve: grava marcos sem LLM extra."""
        blob = f"{user_text or ''} {reply_text or ''}".lower()
        await self.seed_defaults(user_id, character_id)

        rules = [
            (r"\bfacul|faculdade|aula\b", "place_facul", "place", "Faculdade à noite faz parte da rotina dela.", 6),
            (r"\btrabalh", "work", "fact", "Ela trabalha (além de estudar).", 6),
            (r"\bpraia\b", "praia", "place", "Já foram / falam em ir à praia juntos.", 5),
            (r"\bbarzinho\b|\bbar\b", "bar", "place", "Gostam de barzinho.", 5),
            (r"\bsem calcinha\b", "sem_calcinha", "kink_soft", "Já brincaram com ela sair sem calcinha no clima.", 7),
            (r"\bsem suti[aã]\b", "sem_sutia", "kink_soft", "Já brincaram com ela sem sutiã no clima.", 6),
            (r"\bci[uú]me", "ciume_talk", "milestone", "Ciúme dele já foi assunto entre os dois (provocação leve, sem humilhação).", 7),
            (r"\bamigas?\b", "amigas", "person", "Ela tem amigas com quem sai (balada/facul).", 5),
            (r"\bdormi|dormir|pernoite|passar a noite\b", "dormir_junto", "fact", "Eles dormem juntos em algumas noites.", 6),
            (r"\bchurrasco\b", "churrasco", "place", "Já rolou / rola churrasco com flerte ou putaria no meio.", 7),
            (r"\bbalada|festa\b", "balada_festa", "place", "Baladas e festas fazem parte das saídas e aventuras dela.", 7),
            (r"\bshopping\b", "shopping", "place", "Já conheceu / flertou com alguém no shopping.", 7),
            (r"\bcaf[eé]\b", "cafe", "place", "Já rolou encontro/flerte em café.", 6),
            (r"\brestaurante\b", "restaurante", "place", "Já rolou clima ou aventura em restaurante.", 6),
            (r"\bmotel\b", "motel_as_option", "place", "Motel é uma opção (não a única) nas aventuras.", 5),
            (r"\bDP\b|dupla penetra|duas pi[ck]as|dois ao mesmo", "dupla_penetracao_feita", "kink_soft", "Já fez / curte dupla penetração (DP) e pode retomar o assunto.", 9),
            (r"\bbukkake\b", "bukkake_exp", "kink_soft", "Já rolou ou falou de bukkake em alguma aventura.", 7),
            (r"\bgangbang\b|v[aá]rios caras|v[aá]rios homens", "gangbang_exp", "kink_soft", "Gangbang já entrou em alguma aventura (ocasional, não rotina única).", 7),
        ]
        for pat, key, kind, content, imp in rules:
            if re.search(pat, blob, re.I):
                await self.upsert(user_id, character_id, content, kind=kind, key=key, importance=imp)

        # Grava resumo curto da aventura relatada (lugar + nomes se houver)
        if re.search(
            r"\b(trans[aó]|fode|meteu|goz|arromb|amante|motel|balada|festa|"
            r"churrasco|shopping|caf[eé]|restaurante|DP|bukkake|gangbang)\b",
            blob,
            re.I,
        ):
            # nomes masculinos comuns no RP
            names = re.findall(
                r"\b(Marcos|Davi|Kevin|Bruno|Tyrell|Andre|André|Rafael|Diego|"
                r"Pedro|Lucas|Jo[aã]o|Carlos|Felipe|Gabriel|Rafa)\b",
                (reply_text or "") + " " + (user_text or ""),
                re.I,
            )
            place_m = re.search(
                r"\b(motel|balada|festa|churrasco|shopping|caf[eé]|restaurante|"
                r"academia|facul(?:dade)?|uber|carro|praia|bar|casa|after|sauna|airbnb)\b",
                blob,
                re.I,
            )
            place = place_m.group(1) if place_m else "lugar variado"
            who = ", ".join(dict.fromkeys(n.title() for n in names[:4])) or "amante(s)"
            snippet = (reply_text or user_text or "")[:180].replace("\n", " ")
            await self.upsert(
                user_id,
                character_id,
                f"Aventura lembrada ({place} / {who}): {snippet}",
                kind="milestone",
                key=f"aventura_{place.lower()}_{who.lower()[:20].replace(' ', '_')}"[:80],
                importance=8,
            )

        # marco de evento se reply/user menciona danca colada
        if re.search(r"dan[cç]a\s+colad|colad[oa].*cara|cara.*colad", blob, re.I):
            await self.upsert(
                user_id,
                character_id,
                "Marco: dança colada com um cara na balada (ciúme + conversa na cama depois).",
                kind="milestone",
                key="danca_colada",
                importance=9,
            )

    async def recall(
        self,
        user_id: int,
        character_id: int,
        query: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        await self.ensure_table()
        await self.seed_defaults(user_id, character_id)
        lim = limit or self.max_in_context
        r = await self.session.execute(
            text(
                """
                SELECT id, kind, key, content, importance
                FROM long_term_memories
                WHERE user_id=:u AND character_id=:c AND active
                ORDER BY importance DESC, updated_at DESC
                LIMIT :lim
                """
            ),
            {"u": user_id, "c": character_id, "lim": lim},
        )
        rows = [dict(x) for x in r.mappings().all()]
        if query:
            q = query.lower()
            toks = [t for t in re.split(r"\W+", q) if len(t) > 3]

            def score(row):
                c = (row.get("content") or "").lower()
                s = int(row.get("importance") or 0)
                for t in toks:
                    if t in c:
                        s += 2
                return s

            rows.sort(key=score, reverse=True)
        return rows[:lim]

    def format_for_prompt(self, rows: list[dict]) -> str:
        if not rows:
            return "(sem memorias de longo prazo ainda)"
        lines = []
        for r in rows:
            lines.append(
                f"- [{r.get('kind')}|{r.get('key')}] {r.get('content')}"
            )
        return "\n".join(lines)
