"""Workstation -> G1 Jetson internet for INSTALLS ONLY (the robot has no route out): a minimal HTTP
forward proxy (CONNECT + absolute-URI GET), bound to the workstation's 127.0.0.1 and reachable from the robot
solely through g1_internet.sh's `ssh -R`. apt / uv / pip / curl / git all honour http(s)_proxy."""
import asyncio
import sys
from urllib.parse import urlsplit

async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except Exception:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass

async def handle(cr, cw):
    try:
        head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), 30)
        line, *hdrs = head.decode("latin1").split("\r\n")
        method, target, ver = line.split(" ", 2)
        if method == "CONNECT":
            host, port = target.rsplit(":", 1)
            ur, uw = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 20)
            cw.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await cw.drain()
        else:
            u = urlsplit(target)
            path = (u.path or "/") + (("?" + u.query) if u.query else "")
            ur, uw = await asyncio.wait_for(asyncio.open_connection(u.hostname, u.port or 80), 20)
            keep = [h for h in hdrs if h and not h.lower().startswith(("proxy-", "connection:", "keep-alive:"))]
            uw.write((f"{method} {path} {ver}\r\n" + "\r\n".join(keep) + "\r\nConnection: close\r\n\r\n").encode("latin1"))
            await uw.drain()
        print(f"{method} {target[:100]}", flush=True)
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    except Exception as e:
        print(f"ERR {e!r}", flush=True)
        try:
            cw.write(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            await cw.drain()
        except Exception:
            pass
        cw.close()

async def main(port):
    srv = await asyncio.start_server(handle, "127.0.0.1", port)
    print(f"proxy on 127.0.0.1:{port}", flush=True)
    async with srv:
        await srv.serve_forever()

asyncio.run(main(int(sys.argv[1])))
