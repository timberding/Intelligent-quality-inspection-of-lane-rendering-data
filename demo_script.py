# -*- coding: UTF-8 -*-
import os
import torch
import torchvision
import time
import csv

import torch.optim as optim
import torch.nn as nn

from args import build_opt
from loaders import build_loader


def main():
    opt = build_opt()

    model = torchvision.models.resnet50(pretrained=False, num_classes=opt.num_classes).to(opt.device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    loss = nn.CrossEntropyLoss()
    train_loader = build_loader(
        imagedir=opt.traindir,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        metadir=opt.train_metadir,
        metafile=opt.train_metafile,
        require_label=True
    )
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    test_loader = build_loader(
        imagedir=opt.testdir,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        metadir=opt.test_metadir,
        metafile=opt.test_metafile,
        require_label=False
    )

    print('...start training')

    for epoch in range(opt.epochs):
        start_t = time.time()
        epoch_l = 0
        epoch_t = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, label = batch
            image, label = image.to(opt.device), label.to(opt.device)
            output = model(image)

            l = loss(output, label)
            l.backward()
            optimizer.step()

            batch_l = l.item()
            epoch_l += batch_l
            batch_t = time.time() - start_t
            epoch_t += batch_t
            start_t = time.time()
        lr_scheduler.step()
        epoch_t = epoch_t / len(train_loader)
        epoch_l = epoch_l / len(train_loader)
        print('...epoch: {:3d}/{:3d}, loss: {:.4f}, average time: {:.2f}.'.format(
            epoch + 1, opt.epochs, epoch_l, epoch_t))

    print('...start predicting')

    model.eval()
    to_prob = nn.Softmax(dim=1)
    with torch.no_grad():
        imagenames, probs = list(), list()
        for batch_idx, batch in enumerate(test_loader):
            image, imagename = batch
            image = image.to(opt.device)
            pred = model(image)
            prob = to_prob(pred)
            prob = list(prob.data.cpu().numpy())
            imagenames += imagename
            probs += prob

    with open(os.path.join(opt.resultdir, 'submission.csv'), 'w', encoding='utf8') as fp:
        writer = csv.writer(fp)
        writer.writerow(['imagename', 'defect_prob'])
        for info in zip(imagenames, probs):
            imagename, prob = info
            writer.writerow([imagename, str(prob[1])])


if __name__ == "__main__":
    print('.' * 75)
    main()
    print('.' * 75)




