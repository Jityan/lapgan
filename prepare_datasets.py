import argparse
import torch
import torchvision.transforms as transforms

def prepare_dataset(args, split, transform):
    if transform is not None:
        image_transform = transform
    else:
        if split == 'train':
            image_transform = transforms.Compose([
                transforms.Resize(int(args.imsize * 76 / 64)),
                transforms.RandomCrop(args.imsize),
                transforms.RandomHorizontalFlip(),
                ])
        else:
            image_transform = transforms.Compose([
                transforms.Resize((args.imsize, args.imsize)),
                ])
    from datasets import TextImgDataset
    dataset1 = TextImgDataset(split=split, transform=image_transform, args=args)
    from datasets_full import FullTextImgDataset
    dataset2 = FullTextImgDataset(split=split, transform=image_transform, args=args)
    return dataset1, dataset2


def prepare_datasets(args, transform):
    # train dataset
    train_dataset, _ = prepare_dataset(args, split='train', transform=transform)
    # valid dataset
    val_dataset, _ = prepare_dataset(args, split='test', transform=transform)
    # test dataset
    _, test_dataset = prepare_dataset(args, split='test', transform=transform)
    return train_dataset, val_dataset, test_dataset


def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset, test_dataset = prepare_datasets(args, transform)
    # train dataloader
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle=True)
    # valid dataloader
    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, drop_last=True,
            num_workers=num_workers, shuffle=True)
    # valid dataloader
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, drop_last=False,
            num_workers=num_workers, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--data-dir', type=str, default='../data/flowers', help='data path')
    parser.add_argument('--dataset-name', type=str, default='flowers', help='dataset name')
    parser.add_argument('--ch-size', type=int, default=3, help='number of image channels')
    parser.add_argument('--imsize', type=int, default=128, help='image dimensions')
    parser.add_argument('--stage', type=int, default=2, help='number of stages')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    args.clip4text = {'src':"clip", 'type':'ViT-B/32'}

    train_dl, _, _ = prepare_dataloaders(args)
    for step, data in enumerate(train_dl, 0):
        imgs, captions, CLIP_tokens, keys = data
        print(imgs[0].shape, imgs[1].shape, captions, len(CLIP_tokens[0]), keys)
        exit()