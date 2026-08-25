from datetime import datetime, timezone
import random


class DecisionEngine:
    """
    Decide se a personagem manda mensagem proativa.
    Versao companheira: menos restritiva que a original.
    """

    def __init__(self, min_interval_minutes=30, max_daily_messages=12):
        self.min_interval_minutes = max(1, int(min_interval_minutes))
        self.max_daily_messages = max(0, int(max_daily_messages))

    def decide(self, context, autonomy_state, now=None):
        now = now or datetime.now(timezone.utc)

        if not getattr(autonomy_state, "enabled", True):
            return {"action": "wait", "reason": "disabled"}

        quiet = getattr(autonomy_state, "quiet_until", None)
        if quiet and quiet > now:
            return {"action": "wait", "reason": "quiet_until"}

        daily = int(getattr(autonomy_state, "daily_messages", 0) or 0)
        if daily >= self.max_daily_messages:
            return {"action": "wait", "reason": "daily_limit"}

        last_out = getattr(autonomy_state, "last_outbound_at", None)
        if last_out:
            if last_out.tzinfo is None:
                last_out = last_out.replace(tzinfo=timezone.utc)
            elapsed = (now - last_out).total_seconds() / 60
            if elapsed < self.min_interval_minutes:
                return {
                    "action": "wait",
                    "reason": "cooldown",
                    "elapsed_min": round(elapsed, 1),
                }

        relationship = context.get("relationship") or {}
        emotion = context.get("emotion") or {}
        messages = context.get("messages") or []
        memories = context.get("memories") or []
        semantic = context.get("semantic_memories") or []

        closeness = float(relationship.get("closeness", 0) or 0)
        curiosity = float(emotion.get("curiosity", 0) or 0)
        frustration = float(emotion.get("frustration", 0) or 0)
        trust = float(emotion.get("trust", 0) or 0)

        if frustration >= 0.85:
            return {"action": "wait", "reason": "high_frustration"}

        if not messages:
            return {"action": "wait", "reason": "no_conversation"}

        last_user_message = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        if not last_user_message:
            return {"action": "wait", "reason": "no_user_context"}

        created_at = last_user_message.get("created_at")
        if created_at:
            try:
                last_user_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                if last_user_at.tzinfo is None:
                    last_user_at = last_user_at.replace(tzinfo=timezone.utc)
                quiet_after_user = max(10, self.min_interval_minutes * 0.4)
                mins = (now - last_user_at).total_seconds() / 60
                if mins < quiet_after_user:
                    return {
                        "action": "wait",
                        "reason": "recent_user_interaction",
                        "mins_since_user": round(mins, 1),
                    }
            except (ValueError, TypeError):
                pass

        if closeness < 0.05 and len(messages) < 4:
            return {"action": "wait", "reason": "relationship_too_new"}

        score = 0.35
        score += closeness * 0.30
        score += curiosity * 0.20
        score += trust * 0.15
        score += min(0.15, len(memories) * 0.03 + len(semantic) * 0.03)
        score += random.uniform(-0.05, 0.08)

        threshold = 0.40
        if score < threshold:
            return {
                "action": "wait",
                "reason": "low_contextual_interest",
                "score": round(score, 3),
            }

        reason = "missing_you"
        if semantic:
            reason = "contextual_memory"
        elif memories:
            reason = "remembered_user_detail"
        elif curiosity > 0.4:
            reason = "curiosity_and_closeness"

        return {
            "action": "message",
            "reason": reason,
            "score": round(score, 3),
        }
