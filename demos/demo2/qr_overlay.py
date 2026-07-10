import socket

import numpy as np


def guess_lan_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def make_public_url(host, port, public_url=None):
    if public_url:
        return public_url.rstrip("/")
    resolved_host = host
    if host in ("0.0.0.0", "::", ""):
        resolved_host = guess_lan_ip()
    return f"http://{resolved_host}:{int(port)}"


def make_qr_rgb(url, size=220, border=2):
    try:
        import qrcode
    except ImportError as exc:
        raise ImportError(
            "Demo 2 QR display requires qrcode[pil]. Install it with: "
            "python -m pip install 'qrcode[pil]'"
        ) from exc

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=int(border),
    )
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    image = image.resize((int(size), int(size)))
    return np.asarray(image, dtype=np.uint8)
