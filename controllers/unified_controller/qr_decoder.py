import json
import cv2
import numpy as np


def _decode_once(image, method_name):
    detector = cv2.QRCodeDetector()

    result = {
        "qr_decode_success": False,
        "qr_payload": None,
        "qr_points": None,
        "qr_decoder": method_name,
    }

    payload, points, _ = detector.detectAndDecode(image)
    if payload:
        result["qr_decode_success"] = True
        result["qr_payload"] = payload
        if points is not None:
            result["qr_points"] = np.array(points).tolist()
        return result

    ok, decoded_info, points_multi, _ = detector.detectAndDecodeMulti(image)
    if ok and decoded_info:
        payloads = [p for p in decoded_info if p]
        if payloads:
            result["qr_decode_success"] = True
            result["qr_payload"] = payloads[0] if len(payloads) == 1 else payloads
            if points_multi is not None:
                result["qr_points"] = np.array(points_multi).tolist()

    return result


def _parse_compact_payload(payload):
    """
    Supports compact QR payloads like:
        P10294|1A|H
        P10294|1A|L

    Meaning:
        patient_id | bed | urgency_code
    """
    if not isinstance(payload, str):
        return None

    parts = [p.strip() for p in payload.split("|")]
    if len(parts) != 3:
        return None

    patient_id, bed, urgency_code = parts

    urgency_map = {
        "H": "high",
        "L": "low",
    }

    return {
        "patient_id": patient_id,
        "bed": bed,
        "urgency": urgency_map.get(urgency_code.upper(), urgency_code),
    }


def _safe_parse_payload(payload):
    """
    First try JSON.
    If that fails, try compact pipe format.
    """
    if not isinstance(payload, str):
        return None

    try:
        return json.loads(payload)
    except Exception:
        pass

    compact = _parse_compact_payload(payload)
    if compact is not None:
        return compact

    return None


def decode_qr_from_bgr(image_bgr):
    """
    Attempts several QR decoding variants in sequence.

    Returns:
    - qr_decode_success
    - qr_payload
    - qr_points
    - qr_decoder
    - qr_payload_json
    """
    result = {
        "qr_decode_success": False,
        "qr_payload": None,
        "qr_points": None,
        "qr_decoder": None,
        "qr_payload_json": None,
    }

    if image_bgr is None:
        return result

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_up2 = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_up2 = cv2.resize(otsu, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blur_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    variants = [
        ("opencv_raw_bgr", image_bgr),
        ("opencv_gray", gray),
        ("opencv_gray_up2", gray_up2),
        ("opencv_otsu", otsu),
        ("opencv_otsu_up2", otsu_up2),
        ("opencv_blur_otsu", blur_otsu),
    ]

    for method_name, img in variants:
        attempt = _decode_once(img, method_name)
        if attempt["qr_decode_success"]:
            attempt["qr_payload_json"] = _safe_parse_payload(attempt["qr_payload"])
            return attempt

    return result