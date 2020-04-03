""" An example script for using the robusta package.
"""
import robusta
import torch
import torchvision
from torchvision import transforms


# To make this demo run with 0 dependencies and run on any device
# (to show the library features) the demo doesn't use Cuda and the dataset has
# only 1 image.
# After you run it once, feel free to enable Cuda and change the dataset folder


def main():
    model = torchvision.models.resnet50(pretrained=True)

    # Dummy-ImageNetC dataset has only 1 image.
    # Change this to evaluate on a full dataset.
    batch_size = 1
    dataset_folder = 'test/dummy_datasets/ImageNet-C'
    num_epochs = 1

    # We provide implementations for ImageNet-val, ImageNetC, ImageNetR,
    # ImageNetA and ImageNetD:
    val_dataset = robusta.datasets.imagenetc.ImageNetC(
        root=dataset_folder, corruption="gaussian_blur", severity=1,
        transform=transforms.Compose([transforms.ToTensor()])
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    # We offer different options for batch norm adaptation;
    # alternatives are "ema", "batch_wise_prior", ...
    robusta.batchnorm.adapt(model, adapt_type="batch_wise")

    # The accuracy metric can be specific to the dataset:
    # For example, ImageNet-R requires remapping into 200 classes.
    # accuracy_metric = val_dataset.accuracy

    # You can also easily use self-learning in your model.
    # Self-learning adaptation can be combined with batch norm adaptation, example:
    parameters = robusta.selflearning.adapt(model, adapt_type="affine")
    optimizer = torch.optim.SGD(parameters, lr=1e-3)

    # You can choose from a set of adaptation losses (GCE, Entropy, ...)
    rpl_loss = robusta.selflearning.GeneralizedCrossEntropy(q=0.8)

    acc1_sum, acc5_sum, num_samples = 0., 0., 0.
    for epoch in range(num_epochs):
        predictions = []
        for images, targets in val_loader:

            logits = model(images)
            predictions = logits.argmax(dim=1)

            # Predictions are optional. If you do not specify them,
            # they will be computed within the loss function.
            loss = rpl_loss(logits, predictions)

            # When using self-learning, you need to add an additional optimizer
            # step in your evaluation loop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # acc1_sum, acc5_sum += accuracy_metric(predictions, targets, topk=(1,5))
            num_samples += len(targets)
            print(f"Top-1: {acc1_sum/num_samples}, Top-5: {acc5_sum/num_samples}")


if __name__ == '__main__':
    main()
