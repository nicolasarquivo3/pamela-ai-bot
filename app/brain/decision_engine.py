from datetime import datetime, timezone

class DecisionEngine:
    """Conservative, deterministic policy for proactive messages.

    The engine decides *whether* to initiate. The LLM only writes the message
    after the policy has accepted the action.
    """
    def __init__(self, min_interval_minutes=90, max_daily_messages=3):
        self.min_interval_minutes = max(1, int(min_interval_minutes))
        self.max_daily_messages = max(0, int(max_daily_messages))

    def decide(self, context, autonomy_state, now=None):
        now = now or datetime.now(timezone.utc)

        if not autonomy_state.enabled:
            return {"action": "wait", "reason": "disabled"}

        if autonomy_state.quiet_until and autonomy_state.quiet_until > now:
            return {"action": "wait", "reason": "quiet_until"}

        if autonomy_state.daily_messages >= self.max_daily_messages:
            return {"action": "wait", "reason": "daily_limit"}

        if autonomy_state.last_outbound_at:
            elapsed = (now - autonomy_state.last_outbound_at).total_seconds() / 60
            if elapsed < self.min_interval_minutes:
                return {"action": "wait", "reason": "cooldown"}

        relationship = context.get("relationship") or {}
        emotion = context.get("emotion") or {}
        messages = context.get("messages") or []
        memories = context.get("memories") or []
        semantic = context.get("semantic_memories") or []

        closeness = float(relationship.get("closeness", 0))
        curiosity = float(emotion.get("curiosity", 0))
        frustration = float(emotion.get("frustration", 0))
        trust = float(emotion.get("trust", 0))

        if frustration >= 0.70:
            return {"action": "wait", "reason": "high_frustration"}
        if not messages:
            return {"action": "wait", "reason": "no_conversation"}

        last_user_message = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if not last_user_message:
            return {"action": "wait", "reason": "no_user_context"}

        # Do not interrupt a conversation that is still fresh. The same
        # conservative interval used for outbound messages is applied here.
        created_at = last_user_message.get("created_at")
        if created_at:
            try:
                last_user_at = datetime.fromisoformat(created_at)
                if last_user_at.tzinfo is None:
                    last_user_at = last_user_at.replace(tzinfo=timezone.utc)
                if (now - last_user_at).total_seconds() / 60 < self.min_interval_minutes:
                    return {"action": "wait", "reason": "recent_user_interaction"}
            except ValueError:
                pass

        # Do not proactively message users who have not built a relationship.
        if closeness < 0.15:
            return {"action": "wait", "reason": "relationship_too_new"}

        # Require at least one contextual anchor so messages aren't generic spam.
        if not memories and not semantic:
            return {"action": "wait", "reason": "no_context_anchor"}

        score = 0.0
        score += closeness * 0.40
        score += curiosity * 0.25
        score += trust * 0.20
        score += min(0.15, len(memories) * 0.03)

        # Avoid deterministic "yes on every tick": the state changes over time,
        # and this threshold is intentionally conservative.
        threshold = 0.66
        if score < threshold:
            return {"action": "wait", "reason": "low_contextual_interest", "score": round(score, 3)}

        reason = "curiosity_and_closeness"
        if semantic:
            reason = "contextual_memory"
        elif memories:
            reason = "remembered_user_detail"

        return {
            "action": "message",
            "reason": reason,
            "score": round(score, 3),
        }
