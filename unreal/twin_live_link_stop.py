import builtins
link = getattr(builtins, "_twin_live_link", None)
if link:
    link.stop()
    builtins._twin_live_link = None
