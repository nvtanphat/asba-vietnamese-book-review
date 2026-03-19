from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import emoji


DEFAULT_SCAN_PATH = Path("experiments/reports/train_scan.json")
DEFAULT_EMOJI_MAP_PATH = Path("data/maps/emoji_map.json")
DEFAULT_VOCAB_MAP_PATH = Path("data/maps/vocab_map.json")


EMOJI_OVERRIDES = {
    "red_heart": "tim",
    "heart": "tim",
    "sparkling_heart": "tim",
    "two_hearts": "tim",
    "heart_hands": "yêu_thương",
    "smiling_face_with_hearts": "yêu_thích",
    "smiling_face_with_smiling_eyes": "vui_vẻ",
    "smiling_face": "vui_vẻ",
    "grinning_face": "vui_vẻ",
    "grinning_face_with_smiling_eyes": "vui_vẻ",
    "face_with_tears_of_joy": "cười_ra_nước_mắt",
    "rolling_on_the_floor_laughing": "cười_lăn",
    "loudly_crying_face": "buồn_thất_vọng",
    "crying_face": "buồn",
    "pensive_face": "suy_tư",
    "disappointed_face": "thất_vọng",
    "angry_face": "không_hài_lòng",
    "face_with_symbols_on_mouth": "không_hài_lòng",
    "thumbs_up": "tốt",
    "thumbs_down": "không_tốt",
    "ok_hand": "ổn",
    "raised_hands": "vỗ_tay",
    "clapping_hands": "vỗ_tay",
    "folded_hands": "cảm_ơn",
    "thinking_face": "suy_nghĩ",
    "neutral_face": "trung_tính",
    "slightly_smiling_face": "mỉm_cười",
    "smiling_face_with_tear": "cảm_động",
    "sparkles": "lấp_lánh",
    "fire": "nóng",
    "star": "sao",
    "white_check_mark": "đúng",
    "cross_mark": "sai",
    "birthday_cake": "chúc_mừng",
    "wrapped_gift": "quà_tặng",
    "victory_hand": "ổn",
    "person_shrugging": "không_rõ",
    "woman_shrugging": "không_rõ",
    "man_shrugging": "không_rõ",
    "smiling_face_with_halo": "thánh_thiện",
    "face_blowing_a_kiss": "hôn",
    "kissing_face": "hôn",
    "kissing_face_with_smiling_eyes": "hôn",
    "smiling_face_with_heart_eyes": "mê",
    "smiling_face_with_smiling_eyes_and_three_hearts": "yêu_thương",
    "crying_cat": "buồn",
    "grinning_cat": "vui",
    "smiling_cat": "vui",
    "smiling_face_with_sunglasses": "ngầu",
    "sun_with_face": "nắng",
    "sparkling_heart": "tim",
    "green_heart": "tim_xanh",
    "yellow_heart": "tim_vàng",
    "blue_heart": "tim_xanh_dương",
    "purple_heart": "tim_tím",
    "broken_heart": "tan_vỡ",
    "100": "trăm_điểm",
}


VOCAB_OVERRIDES = {
    "ko": "không",
    "k": "không",
    "dc": "được",
    "đc": "được",
    "j": "gì",
    "r": "rồi",
    "vs": "với",
    "mn": "mọi_người",
    "mng": "mọi_người",
    "sp": "sản_phẩm",
    "hok": "không",
    "kh": "không",
    "bt": "bình_thường",
    "bth": "bình_thường",
    "ib": "inbox",
    "cmt": "comment",
    "cod": "thanh_toán_khi_nhận_hàng",
    "ship": "giao_hàng",
    "sdt": "số_điện_thoại",
    "sr": "xin_lỗi",
    "tk": "tài_khoản",
    "fs": "free_ship",
    "vc": "voucher",
    "mk": "mình",
    "ntn": "như_thế_nào",
    "trc": "trước",
    "nxb": "nhà_xuất_bản",
    "cx": "cũng",
    "nd": "nội_dung",
    "ms": "mới",
    "nv": "nhân_viên",
    "vd": "ví_dụ",
    "nx": "nhận_xét",
    "dth": "điện_thoại",
    "bh": "bảo_hành",
    "km": "khuyến_mãi",
    "cs": "chăm_sóc",
    "hcm": "hồ_chí_minh",
    "sgk": "sách_giáo_khoa",
    "ql": "quản_lý",
    "kbt": "không_biết",
    "tks": "cảm_ơn",
    "hsg": "học_sinh_giỏi",
    "trn": "trên",
    "cv": "công_việc",
    "nhx": "nhận_xét",
    "thw": "thương",
    "vtp": "viettel_post",
    "gq": "giải_quyết",
    "gr": "group",
    "tl": "trả_lời",
    "nhg": "nhưng",
    "dk": "đăng_ký",
}

DOMAIN_OVERRIDES = {
    "bookcare": "book_care",
    "freeship": "free_ship",
    "happylive": "happy_live",
    "mcbooks": "mcbooks",
    "goodreads": "goodreads",
    "fahasa": "fahasa",
    "shopee": "shopee",
    "shipper": "shipper",
    "bookmark": "bookmark",
    "ebook": "ebook",
    "google": "google",
    "app": "app",
}


def _demojize_alias(raw_emoji: str) -> list[str]:
    demojized = emoji.demojize(raw_emoji, language="en")
    return re.findall(r":([^:]+):", demojized)


def _normalize_alias(alias: str) -> str:
    alias = alias.lower()
    alias = re.sub(r"[^a-z0-9]+", "_", alias)
    return alias.strip("_")


def build_emoji_map(report: dict[str, Any]) -> dict[str, str]:
    emoji_rows = report["checks"]["emoji"]["all_emojis"]
    mapping: dict[str, str] = dict(EMOJI_OVERRIDES)
    for item in emoji_rows:
        raw = item["emoji"]
        for alias in _demojize_alias(raw):
            key = _normalize_alias(alias)
            value = EMOJI_OVERRIDES.get(key)
            if value is None:
                base_alias = _normalize_alias(
                    re.sub(
                        r"_(?:light|medium_light|medium|medium_dark|dark)_skin_tone$",
                        "",
                        alias,
                    )
                )
                value = EMOJI_OVERRIDES.get(base_alias, f"emoji_{base_alias}")
            mapping[key] = value
    return dict(sorted(mapping.items()))


def build_vocab_map(report: dict[str, Any]) -> dict[str, str]:
    vocab_rows = report["checks"]["vocab"]["teencode_like_tokens"]
    mapping: dict[str, str] = dict(VOCAB_OVERRIDES)
    for item in vocab_rows:
        token = item["token"]
        mapping[token] = VOCAB_OVERRIDES.get(token.lower(), token)
    for token, value in DOMAIN_OVERRIDES.items():
        mapping[token] = value
    return dict(sorted(mapping.items()))


def write_json(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_maps(scan_path: Path, emoji_map_path: Path, vocab_map_path: Path) -> None:
    report = json.loads(scan_path.read_text(encoding="utf-8"))
    write_json(emoji_map_path, build_emoji_map(report))
    write_json(vocab_map_path, build_vocab_map(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build preprocessing maps from a scan report.")
    parser.add_argument("--scan", default=str(DEFAULT_SCAN_PATH))
    parser.add_argument("--emoji-map", default=str(DEFAULT_EMOJI_MAP_PATH))
    parser.add_argument("--vocab-map", default=str(DEFAULT_VOCAB_MAP_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_maps(Path(args.scan), Path(args.emoji_map), Path(args.vocab_map))
    print(args.emoji_map)
    print(args.vocab_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
