def confirm(prompt):
    reply = input(f"{prompt} ").strip().lower()
    return reply in {"y", "yes"}


def pause(prompt):
    input(f"{prompt}\n> ")


__all__ = ["confirm", "pause"]
