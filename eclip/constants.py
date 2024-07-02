CXR14_LABELS = [
    "No_Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
CXR14x100_LABELS = CXR14_LABELS[:-1]
CXR8_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]
CXR14_TO_CXR8_MAPPING = {
    CXR14_LABELS.index(label): CXR8_LABELS.index(label) for label in CXR8_LABELS
}

CHEXPERT8_20_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Edema",
    "Fracture",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
]

CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


RSNA_CLASS_PROMPTS = {
    "Pneumonia": {
        "findings": [
            "Evidence of consolidation",
            "Patchy opacities noted",
            "Air space opacification",
            "Lobar consolidation",
            "Bronchopneumonia pattern",
            "Interstitial infiltrates",
            "Alveolar infiltrates",
            "Ground-glass opacity",
            "Pulmonary infiltrates",
            "Peribronchial thickening",
            "Bilateral infiltrates",
            "Focal shadowing",
        ],
        "severity": [
            "Mild interstitial changes",
            "Moderate alveolar consolidation",
            "Severe bilateral pneumonia",
            "Extensive disease burden",
        ],
        "additional_descriptors": [
            "with possible pleural effusion",
            "suggestive of bacterial infection",
            "may indicate viral etiology",
            "consistent with aspiration pneumonia",
            "with signs of organizing pneumonia",
            "lobar pneumonia present",
            "indicative of community-acquired pneumonia",
            "compatible with hospital-acquired pneumonia",
            "",
        ],
    },
    "No Pneumonia": {
        "normal_findings": [
            "No evidence of acute disease",
            "Lungs are clear",
            "No focal consolidation",
            "No signs of infection",
            "Clear lung fields",
            "No pleural effusion or pneumothorax",
            "Normal lung parenchyma",
            "No acute cardiopulmonary abnormality",
        ]
    },
}
