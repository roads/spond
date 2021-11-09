# Given a co-occurrence matrix, and a file of labels to names,
# and a file of the universe of labels to names,
# test that the correct indexes are included in the output
import sys

from spond.experimental.glove.aligned_glove import DataDictionary

sys.path.append("/opt/github.com/spond")

from io import StringIO
from unittest import TestCase
import torch

# all labels: These will be numbered 0 to 7

ALL_LABELS = f"""
#LabelName,DisplayName
/m/09x0r,"Speech"
/m/05zppz,"Male person"
/m/02zsn,"Female person"
/m/0ytgt,"Child"
/m/01h8n0,"Conversation"
/m/02qldy,"Narration, monologue"
/m/0261r1,"Babbling"
/m/0brhx,"Speech synthesizer"
"""

ALL_LABELS_INDEXES = list(range(8))

X_LABELS = f"""
mid,display_name
/m/09x0r,"Speech"
/m/05zppz,"Male person"
/m/02zsn,"Female person"
/m/0ytgt,"Child"
/m/01h8n0,"Conversation"
"""

X_LABELS_INDEXES = [0, 1, 2, 3, 4]

X_COOC = torch.tensor([
    [0.0, 2, 3, 3, 1],
    [2,   0, 4, 0, 3],
    [3,   4, 0, 2, 2],
    [3,   0, 2, 0, 1],
    [1,   3, 2, 1, 0],
])

Y_LABELS = f"""
mid,display_name
/m/05zppz,"Male person"
/m/02zsn,"Female person"
/m/0ytgt,"Child"
/m/02qldy,"Narration, monologue"
/m/0brhx,"Speech synthesizer"
"""

Y_LABELS_INDEXES = [1, 2, 3, 5, 7]

Y_COOC = torch.tensor([
    [0.0, 1, 5, 3, 1],
    [1,   0, 2, 0, 3],
    [5,   2, 0, 2, 1],
    [3,   0, 2, 0, 1],
    [1,   3, 1, 1, 0],
])

# first item: index in all_labels
# second item: index in x
# second item: index in y
INTERSECTION_INDEXES = [
    [1, 1, 0],
    [2, 2, 1],
    [3, 3, 2]
]


class TestCoocIndexParsing(TestCase):

    def setUp(self):
        self.all_labels = StringIO(ALL_LABELS.strip())
        self.all_labels_indexes = torch.tensor(ALL_LABELS_INDEXES)
        self.x_labels = StringIO(X_LABELS.strip())
        self.x_labels_indexes = torch.tensor(X_LABELS_INDEXES)
        self.y_labels = StringIO(Y_LABELS.strip())
        self.y_labels_indexes = torch.tensor(Y_LABELS_INDEXES)
        self.intersection_indexes = torch.tensor(INTERSECTION_INDEXES)

    def test(self):
        # Test that the data dictionary does the right thing to calculate
        # union and intersection of indexes
        dd = DataDictionary(
            torch.tensor(X_COOC),      # co-occurrence matrix for x
            self.x_labels,    # full filepath or handle containing all x labels to names
            torch.tensor(Y_COOC),      # co-occurrence matrix for y
            self.y_labels,    # full filepath or handle containing all y labels to names
            self.all_labels,
        )

        self.assertTrue((dd.x_indexes == self.x_labels_indexes).all().item())
        self.assertTrue((dd.y_indexes == self.y_labels_indexes).all().item())
        self.assertTrue((dd.intersection_indexes == self.intersection_indexes).all().item())
