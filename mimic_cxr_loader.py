import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

"""
Read file to obtain its raw string contents.
"""
def read_file(src_directory, filename, encoding=None, log=True):
    if log: print("Reading file:", filename)
    with open(os.path.join(src_directory, filename), mode='r', encoding=encoding) as file:
        content = file.read().split("\n")
    return content


"""
Data loader for Mimic-CXR medical images dataset.
"""
class MimicCXRLoader(Dataset):
    def __init__(self, root, split):
        # Initialize variables
        self.root = root if root is not None \
            else "/home/geomos/Documents/MIMIC/rrs-mimic-cxr"
        self.findings = []
        self.impression = []
        self.images = []

        # Process the patients information
        findings = read_file(self.root, split + ".findings.tok")
        impression = read_file(self.root, split + ".impression.tok")
        patients = read_file(self.root, split + ".image.tok")
        for i in tqdm(range(len(patients))):
            patient_case = patients[i]
            images = patient_case.split(",")
            for j in range(len(images)):
                self.findings.append(findings[i])
                self.impression.append(impression[i])
                self.images.append(images[j])

class MimicCXRImageLoader(MimicCXRLoader):
    def __init__(self, root=None, split='train'):
        super(MimicCXRImageLoader, self).__init__(root, split)

    def __getitem__(self, index):
        # Image normalization
        image = Image.open(os.path.join(self.root, self.images[index]))
        image = image.convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)
        caption = self.findings[index]

        return image, caption


class MimicCXRTextLoaderDuplicated(MimicCXRLoader):
    def __init__(self, root=None, split='train'):
        super(MimicCXRTextLoaderDuplicated, self).__init__(root, split)

    def __getitem__(self, index):
        in_caption = self.impression[index]
        out_caption = self.findings[index]

        return in_caption, out_caption

    def __len__(self):
        return len(self.images)


class MimicCXRTextLoaderUnique(Dataset):
    def __init__(self, root, split):
        # Initialize variables
        self.root = root if root is not None \
            else "/home/geomos/Documents/MIMIC/rrs-mimic-cxr"
        self.findings = []
        self.impression = []

        # Process the patients information
        findings = read_file(self.root, split + ".findings.tok")
        impression = read_file(self.root, split + ".impression.tok")
        for i in tqdm(range(len(findings))):
            self.findings.append(findings[i])
            self.impression.append(impression[i])

    def __getitem__(self, index):
        in_caption = self.impression[index]
        out_caption = self.findings[index]

        return in_caption, out_caption

    def __len__(self):
        return len(self.images)