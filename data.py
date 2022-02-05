import torch
from torch.utils.data import Dataset , Sampler, RandomSampler
from PIL import Image
import os
import copy
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from numpy.random import choice

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
class ContrastiveDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        self.classes = {"No findings":[]}
        for i in CLASS_NAMES:
            self.classes[i] = []
        with open(split, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
                if(sum(label)==0):
                    self.classes["No findings"].append(image_name)
                for l in range(len(label)):
                    if(label[l] == 1):
                        self.classes[CLASS_NAMES[l]].append(image_name)
        print("Dataset summery")
        for i,j in self.classes.items():
            print(i,len(j))
        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('L')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
    def _get_class(self,cls):
        return self.classes[cls]
    def _get_lengths(self):
        return {y:len(x) for y,x in self.classes.items()}

class ContrastiveRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            yield from torch.randperm(n, generator=generator).tolist()

    def __len__(self) -> int:
        return self.num_samples


class ContrastiveBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, batch_size: int, drop_last: bool,dataset: ContrastiveDataset) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        # self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset
        self.samplers = [ContrastiveRandomSampler(self.dataset._get_class("No findings"))]
        for c in range(len(CLASS_NAMES)):
            self.samplers.append(ContrastiveRandomSampler(self.dataset._get_class(CLASS_NAMES[c]),replacement= True,num_samples = len(self.dataset._get_class("No findings"))))

    def __iter__(self) -> Iterator[List[int]]:
        draw = choice(self.samplers, self.batch_size, replace=False, p=([0.99]+[0.01/(len(self.samplers)-1)]*(len(self.samplers)-1)))
        batch = []
        it = []
        for samp in self.samplers:
            it.append(iter(samp))
        print(it[0])
        print(it)
        for _ in range(len(self.samplers[0])):
            for j in draw:
                batch.append(next(it[j]))
            for k in draw:
                batch.append(next(it[k]))
            yield batch
            batch = []
            draw = choice(self.samplers, self.batch_size, replace=False, p=([0.99]+[0.01/(len(self.samplers)-1)]*(len(self.samplers)-1)))
 

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]

        t_len = len(self.samplers[0])*len(self.samplers)
        # for s in self.samplers:
        #     t_len += len(s)
        return t_len
        # if self.drop_last:
        #     return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        # else:
        #     return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
