import socket

import numpy as np


TRAVEL_ROUTER_WIFI_SSID = "Emacs"
TRAVEL_ROUTER_WIFI_PASSWORD = "315810612"
TRAVEL_ROUTER_WIFI_SECURITY = "WPA"


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


def _escape_wifi_qr_value(value):
    escaped = str(value).replace("\\", "\\\\")
    for character in (";", ",", ":", '"'):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def make_travel_router_wifi_payload():
    """Build the standard phone-scannable Wi-Fi QR payload."""
    security = _escape_wifi_qr_value(TRAVEL_ROUTER_WIFI_SECURITY)
    ssid = _escape_wifi_qr_value(TRAVEL_ROUTER_WIFI_SSID)
    password = _escape_wifi_qr_value(TRAVEL_ROUTER_WIFI_PASSWORD)
    return f"WIFI:T:{security};S:{ssid};P:{password};;"


def _load_qr_label_font(font_size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", int(font_size))
    except OSError:
        try:
            return ImageFont.load_default(size=int(font_size))
        except TypeError:
            return ImageFont.load_default()


def _draw_centered_label(draw, text, center_x, y, font, fill):
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (int(center_x - width / 2), int(y - bounds[1] - height / 2)),
        text,
        font=font,
        fill=fill,
    )


def _make_labeled_qr_rgb(payload, label, size, border, footer_lines=()):
    from PIL import Image, ImageDraw

    size = int(size)
    footer_lines = tuple(str(line) for line in footer_lines)
    label_height = max(28, int(round(size * 0.1125)))
    footer_font_size = max(12, int(round(size * 0.053)))
    footer_line_height = footer_font_size + max(2, size // 100)
    footer_line_gap = max(3, size // 100)
    footer_padding = max(6, size // 40)
    footer_height = 0
    if footer_lines:
        footer_height = (
            2 * footer_padding
            + len(footer_lines) * footer_line_height
            + (len(footer_lines) - 1) * footer_line_gap
        )
    panel_width = size
    panel_height = label_height + size + footer_height
    panel = Image.new("RGB", (panel_width, panel_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(panel)
    label_font = _load_qr_label_font(max(15, int(round(size * 0.06875))))
    label_color = (0, 220, 255)
    label_y = label_height // 2

    _draw_centered_label(
        draw,
        label,
        size // 2,
        label_y,
        label_font,
        label_color,
    )
    qr = make_qr_rgb(payload, size=size, border=border)
    panel.paste(Image.fromarray(qr), (0, label_height))

    if footer_lines:
        footer_font = _load_qr_label_font(footer_font_size)
        footer_color = (255, 215, 0)
        footer_y = label_height + size + footer_padding + footer_line_height // 2
        for line in footer_lines:
            _draw_centered_label(
                draw,
                line,
                size // 2,
                footer_y,
                footer_font,
                footer_color,
            )
            footer_y += footer_line_height + footer_line_gap
    return np.array(panel, dtype=np.uint8, copy=True)


def make_public_display_qr_overlays(
    url,
    size=220,
    border=2,
    include_travel_router_wifi=False,
):
    """Return ``(controller_qr, wifi_qr)`` for the selected connection mode.

    The default is a single controller QR. Explicit travel-router mode adds a
    numbered Wi-Fi overlay so the renderer can place Wi-Fi at the far-left edge
    and the controller at the far-right edge without guessing from a LAN IP.
    """
    if not include_travel_router_wifi:
        return make_qr_rgb(url, size=size, border=border), None

    wifi_qr = _make_labeled_qr_rgb(
        make_travel_router_wifi_payload(),
        f"1  JOIN {TRAVEL_ROUTER_WIFI_SSID} WI-FI",
        size,
        border,
        footer_lines=(
            "IF ALREADY ON ANOTHER WI-FI",
            "DISCONNECT, THEN SCAN",
        ),
    )
    controller_qr = _make_labeled_qr_rgb(
        url,
        "2  OPEN CONTROLLER",
        size,
        border,
    )
    return controller_qr, wifi_qr
