import streamlit as st
import json
from google.cloud import firestore
from google.oauth2 import service_account
from collections import defaultdict
import numpy as np

fb_credentials = st.secrets["firebase"]
creds = service_account.Credentials.from_service_account_info(fb_credentials)
db = firestore.Client(credentials=creds, project="feature-annotation")

# Create a reference to the Google post.
# doc_ref = db.collection("stats").document("_test")

# Then get the data at that reference.
# doc = doc_ref.get()

# Let's see what we got!
# st.write("The id is: ", doc.id)
# st.write("The contents are: ", doc.to_dict().keys())
# doc_ref.set({"count": 3})


# create new document
# load sparse-dense_random-RC_contexts.json
with open("sparse-dense_random-RC_contexts.json", "r") as f:
    data = json.load(f)
    for i in data.keys():
        doc_ref = db.collection("stats").document(str(i))
        doc_ref.set({
            "annotation_count": 0,
        })

# # Find a random document with the least annotations
# stats = db.collection("stats").stream()
# cnt_dict = defaultdict(list)
# total_annotations = 0
# for stat in stats:
#     cnt = stat.to_dict()["annotation_count"]
#     total_annotations += cnt
#     cnt_dict[cnt].append(stat.id)
# least_annotated = min(cnt_dict.keys())
# random_id = np.random.choice(cnt_dict[least_annotated])