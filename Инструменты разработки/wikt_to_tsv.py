"""tsnorm wordforms.dat (pickle, только данные) -> TSV: ключ \t форма с «+» \t теги."""
import gzip, pickle, sys
VOW = "аеёиоуыэюя"
class Safe(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"запрещено: {module}.{name}")
def marked(form, pos):
    pos = sorted(pos if isinstance(pos, list) else [pos]); out = list(form.lower())
    for i in reversed(pos):
        if 0 <= i < len(out) and out[i] in VOW: out.insert(i, "+")
    return "".join(out)
SHORT = {"nominative":"им","genitive":"род","dative":"дат","accusative":"вин","instrumental":"тв",
         "prepositional":"пр","locative":"мест","singular":"ед","plural":"мн","present":"наст",
         "past":"прош","future":"буд","canonical":"осн","masculine":"м","feminine":"ж","neuter":"ср",
         "short":"кр","comparative":"сравн","imperative":"пов","participle":"прич","adverbial":"дееп"}
forms = Safe(open(sys.argv[1], "rb")).load()
n = 0
with gzip.open(sys.argv[2], "wt", encoding="utf-8", compresslevel=9) as f:
    f.write("# Ударения словоформ: Викисловарь (CC BY-SA 3.0), через tsnorm 1.1.2 (MIT). ключ\tформа\tтеги\n")
    for word, items in sorted(forms.items()):
        if " " in word or not isinstance(items, list):
            continue
        k = word.lower().replace("ё", "е")
        if sum(c in VOW for c in k) < 1:
            continue
        acc = {}
        for it in items:
            if not isinstance(it, dict): continue
            v = marked(it.get("word_form") or word, it.get("stress_pos") or [])
            acc.setdefault(v, set()).update(SHORT[t] for t in str(it.get("form_tags") or "").split() if t in SHORT)
        for v, tags in sorted(acc.items()):
            f.write(f"{k}\t{v}\t{' '.join(sorted(tags))}\n"); n += 1
print("строк:", n)
