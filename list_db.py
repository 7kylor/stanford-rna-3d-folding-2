
from rna_tbm.template_db import TemplateDB
db = TemplateDB.load("output/template_db.pkl")
print(f"Total pdb_ids: {len(db.templates)}")
print(f"First 10 pdb_ids: {list(db.templates.keys())[:10]}")
