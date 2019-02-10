import matplotlib.pyplot as plt
from simple import load_training_data as simple_h
from LeNet import load_training_data as LeNet_h


def comparison_plot(simple_history, lenet_history):
    """
    :param simple_history: training data (dict)
    :param lenet_history: training data (dict)
    :return: matplotlib-.pyplot.show()
    ---------------------------
    Plots a comparison of train/validation
    accuracy and loss between the simple
    and the LeNet model.
    Saved as comparison.png
    ---------------------------
    """
    # Generate figure and axes
    fig, ax = plt.subplots(2, 2)
    # Subplot 1: training accuracy.
    ax0 = ax[0, 0]
    ax0.set_title("Training accuracy")
    ax0.plot(simple_history["acc"])
    ax0.plot(lenet_history["acc"])
    ax0.set_ylabel("accuracy")
    ax0.set_xlabel("epoch")
    ax0.legend(["simple", "LeNet"])
    ax0 = plt.gca()
    ax0.set_xlim([0, 10])
    ax0.set_ylim([0, 1])
    # Subplot 2: validation accuracy.
    ax1 = ax[0, 1]
    ax1.set_title("Validation accuracy")
    ax1.plot(simple_history["val_acc"])
    ax1.plot(lenet_history["val_acc"])
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend(["simple", "LeNet"])
    ax1 = plt.gca()
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 1])
    # Subplot 3: training loss.
    ax2 = ax[1, 0]
    ax2.set_title("Training loss")
    ax2.plot(simple_history["loss"])
    ax2.plot(lenet_history["loss"])
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend(["simple", "LeNet"])
    ax2 = plt.gca()
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 1])
    # Subplot 4: validation loss.
    ax3 = ax[1, 1]
    ax3.set_title("Validation loss")
    ax3.plot(simple_history["val_loss"])
    ax3.plot(lenet_history["val_loss"])
    ax3.set_ylabel("accuracy")
    ax3.set_xlabel("epoch")
    ax3.legend(["simple", "LeNet"])
    ax3 = plt.gca()
    ax3.set_xlim([0, 10])
    ax3.set_ylim([0, 1])
    # Tighten and sava figure.
    plt.tight_layout()
    plt.savefig("comparison.png")
    return plt.show()


def main():
    """
    Comparison plot:
    ---------------------------
    - Imports load_training_data() from from both simple.py and LeNet.py.
    - Imports training history for the simple and the LeNet-5 models.
    - Plots a comparison between the two models.
    ---------------------------
    """
    # File names.
    simple_h_name = "simple_train_hist_dict"
    lenet_h_name = "lenet_train_hist_dict"

    # Import load_training_data()
    simple_hist = simple_h(simple_h_name)
    lenet_hist = LeNet_h(lenet_h_name)

    # Plot a comparison
    comparison_plot(simple_hist, lenet_hist)


if __name__ == "__main__":
    main()
