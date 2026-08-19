"""
generate_cf_news.py
Offline script that reads news.tsv and generates:
  - news_cf.tsv        : counterfactual articles (same format as news.tsv)
  - news_cf_map.json   : {original_news_id -> cf_news_id} for articles that have a cf version

Only articles from ELIGIBLE_CATEGORIES with at least one entity in OPPOSITE_ENTITY_MAP
(and of ELIGIBLE_TYPES) are augmented. All others are skipped.
"""

import json
import os
import re

from opposite_entities import OPPOSITE_ENTITY_MAP, ELIGIBLE_CATEGORIES, ELIGIBLE_TYPES


def load_wikidata_labels(news_tsv_path):
    """Build a {WikidataId -> Label} map from news.tsv entity annotations."""
    label_map = {}
    with open(news_tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            for field in [parts[6], parts[7]]:
                try:
                    for e in json.loads(field):
                        if e['WikidataId'] not in label_map:
                            label_map[e['WikidataId']] = e['Label']
                except Exception:
                    pass
    return label_map


def replace_surface_form(text, surface_forms, new_label):
    """Replace the first matching surface form in text with new_label (case-insensitive)."""
    for sf in surface_forms:
        pattern = re.compile(re.escape(sf), re.IGNORECASE)
        new_text, count = pattern.subn(new_label, text, count=1)
        if count > 0:
            return new_text
    return text  # no replacement found


def generate_cf_news(input_tsv, output_tsv, output_map_json, label_map):
    """
    Reads input_tsv (news.tsv format), writes output_tsv and output_map_json.
    Returns (total_processed, total_augmented).
    """
    cf_map = {}
    augmented = 0
    total = 0

    with open(input_tsv, 'r', encoding='utf-8') as fin, \
         open(output_tsv, 'w', encoding='utf-8') as fout:

        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            news_id   = parts[0]
            category  = parts[1]
            subcategory = parts[2]
            title     = parts[3]
            abstract  = parts[4]
            url       = parts[5]
            title_ent_raw    = parts[6]
            abstract_ent_raw = parts[7]

            total += 1

            # Only augment eligible categories
            if category not in ELIGIBLE_CATEGORIES:
                continue

            # Parse entities from both title and abstract
            try:
                title_ents    = json.loads(title_ent_raw)
            except Exception:
                title_ents = []
            try:
                abstract_ents = json.loads(abstract_ent_raw)
            except Exception:
                abstract_ents = []

            all_ents = title_ents + abstract_ents

            # Find the first entity eligible for swapping
            swap_entity = None
            for e in all_ents:
                if e['Type'] in ELIGIBLE_TYPES and e['WikidataId'] in OPPOSITE_ENTITY_MAP:
                    swap_entity = e
                    break

            if swap_entity is None:
                continue  # no swappable entity — skip this article

            orig_wid      = swap_entity['WikidataId']
            opp_wid       = OPPOSITE_ENTITY_MAP[orig_wid]
            opp_label     = label_map.get(opp_wid, orig_wid)  # fallback to wikidata id if label unknown
            surface_forms = swap_entity.get('SurfaceForms', [swap_entity['Label']])

            # Replace surface form in title and abstract text
            cf_title    = replace_surface_form(title,    surface_forms, opp_label)
            cf_abstract = replace_surface_form(abstract, surface_forms, opp_label)

            # Update entity JSON: swap the entity entry
            def swap_ent_json(ent_list):
                new_list = []
                swapped = False
                for e in ent_list:
                    if not swapped and e['WikidataId'] == orig_wid:
                        new_e = dict(e)
                        new_e['WikidataId'] = opp_wid
                        new_e['Label']      = opp_label
                        new_e['SurfaceForms'] = [opp_label]
                        new_e['Confidence'] = 1.0
                        new_list.append(new_e)
                        swapped = True
                    else:
                        new_list.append(e)
                return new_list

            cf_title_ents    = swap_ent_json(title_ents)
            cf_abstract_ents = swap_ent_json(abstract_ents)

            # Assign cf news_id
            cf_news_id = news_id + '_cf'

            # Write cf article row (same TSV format)
            cf_line = '\t'.join([
                cf_news_id,
                category,
                subcategory,
                cf_title,
                cf_abstract,
                url,
                json.dumps(cf_title_ents,    ensure_ascii=False),
                json.dumps(cf_abstract_ents, ensure_ascii=False),
            ])
            fout.write(cf_line + '\n')

            cf_map[news_id] = cf_news_id
            augmented += 1

    # Write mapping file
    with open(output_map_json, 'w', encoding='utf-8') as fmap:
        json.dump(cf_map, fmap, ensure_ascii=False, indent=2)

    return total, augmented


def run(dataset_dir):
    train_dir = os.path.join(dataset_dir, 'MINDsmall_train')
    input_tsv      = os.path.join(train_dir, 'news.tsv')
    output_tsv     = os.path.join(train_dir, 'news_cf.tsv')
    output_map     = os.path.join(train_dir, 'news_cf_map.json')

    if os.path.exists(output_tsv) and os.path.exists(output_map):
        print('[generate_cf_news] news_cf.tsv and news_cf_map.json already exist — skipping generation.')
        return

    print('[generate_cf_news] Loading Wikidata label map...')
    label_map = load_wikidata_labels(input_tsv)

    print('[generate_cf_news] Generating counterfactual news...')
    total, augmented = generate_cf_news(input_tsv, output_tsv, output_map, label_map)

    print(f'[generate_cf_news] Done. {augmented}/{total} articles augmented.')
    print(f'[generate_cf_news] Output: {output_tsv}')
    print(f'[generate_cf_news] Map:    {output_map}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    args = parser.parse_args()
    run(args.dataset_dir)
