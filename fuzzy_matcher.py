import re

# Map of equivalent/confusable OCR characters
OCR_EQUIVALENTS = {
    '0': 'O', 'O': '0',
    '8': 'B', 'B': '8',
    '1': 'I', 'I': '1', 'T': '1',
    '5': 'S', 'S': '5',
    '2': 'Z', 'Z': '2',
    '4': 'A', 'A': '4',
    '6': 'G', 'G': '6'
}

def levenshtein_distance(s1, s2):
    """Calculates standard Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def ocr_weighted_distance(s1, s2):
    """
    Calculates weighted Levenshtein distance, treating known OCR character swaps 
    (e.g., 0/O, 8/B, 1/I) with a reduced penalty of 0.2 instead of 1.0.
    """
    s1 = re.sub(r'[^A-ZA-Z0-9]', '', s1.upper())
    s2 = re.sub(r'[^A-ZA-Z0-9]', '', s2.upper())

    if s1 == s2:
        return 0.0

    if abs(len(s1) - len(s2)) > 1:
        return float(abs(len(s1) - len(s2)))

    length = max(len(s1), len(s2))
    if length == 0:
        return 0.0

    dist = 0.0
    i = j = 0
    while i < len(s1) and j < len(s2):
        c1 = s1[i]
        c2 = s2[j]
        if c1 == c2:
            pass
        elif OCR_EQUIVALENTS.get(c1) == c2 or OCR_EQUIVALENTS.get(c2) == c1:
            dist += 0.2  # Minor penalty for OCR typo
        else:
            dist += 1.0  # Standard substitution penalty
        i += 1
        j += 1

    dist += abs(len(s1) - len(s2)) * 1.0
    return round(dist, 2)

def fuzzy_match_plate(ocr_plate, blacklist_records, max_distance=1.2):
    """
    Checks ocr_plate against a list of blacklisted records.
    Returns (is_match, matched_record, confidence, distance)
    """
    ocr_clean = re.sub(r'[^A-ZA-Z0-9]', '', ocr_plate.upper())
    if not ocr_clean or not blacklist_records:
        return False, None, 0.0, 999.0

    best_match = None
    min_dist = 999.0

    for record in blacklist_records:
        target_plate = record["plate_number"]
        dist = ocr_weighted_distance(ocr_clean, target_plate)

        if dist < min_dist:
            min_dist = dist
            best_match = record

    if min_dist <= max_distance and best_match is not None:
        target_len = len(best_match["plate_number"])
        confidence = round(max(0.70, 1.0 - (min_dist / float(target_len))), 2)
        return True, best_match, confidence, min_dist

    return False, None, 0.0, min_dist
