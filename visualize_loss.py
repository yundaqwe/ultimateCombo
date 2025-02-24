import matplotlib.pyplot as plt
import pandas as pd
from ipdb import set_trace
from pandas import Series, DataFrame
import  numpy as np


def calculate_cdf(data,file):
    sorted_random_data = np.sort(data)
    p = 1. * np.arange(len(sorted_random_data)) / float(len(sorted_random_data) - 1)
    # fig = plt.figure()
    # fig.suptitle('CDF of data points')
    # ax2 = fig.add_subplot(111)
    # ax2.plot(sorted_random_data, p)
    # ax2.set_xlabel('sorted_random_data')
    # ax2.set_ylabel('p')
    # plt.savefig(f"./loss/{file[:16]}_describe_loss_cdf.jpg")
    return p,sorted_random_data
def draw_pdf(modified_file,file):
    bins = 100
    # plt.figure(figsize=(10,20))
    fig, axes = plt.subplots(4, 3,figsize=(12, 12))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    axes=axes.flatten()
    for i,ax in enumerate(axes):
        if i==11:
            continue
        # set_trace()
        x=modified_file[include_methods[i]]
        n, bins, patches = ax.hist(x, bins, density=True, histtype="bar", facecolor="#99FF33", edgecolor="#00FF99",
                                   alpha=0.75)
        p,sorted_random_data=calculate_cdf(x, file)
        ax1 = ax.twinx()
        ax1.plot(sorted_random_data, p)
        # y = ((1 / (np.power(2 * np.pi, 0.5) * sigma)) * np.exp(-0.5 * np.power((bins - mu) / sigma, 2)))

        # ax.plot(bins, y, color="#7744FF", ls="--", lw=2)

        ax.grid(ls=":", lw=1, color="gray", alpha=0.2)
        # ax.text(54, 0.2, r"$y=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$",
        #         {"color": "#FF5511", "fontsize": 20})
        #
        ax.set_xlabel(f"{include_methods[i]}")
        ax.set_ylabel("PDF")
        ax1.set_ylabel("CDF")

    fig.tight_layout()
    plt.show()
    plt.savefig(f"./loss/{file}_describe_loss.jpg", dpi=600)

list_nets = [
    'tf_inception_v3',
    'tf_inception_v4',
    'tf_resnet_v2_50',
    'tf_resnet_v2_101',
    'tf_resnet_v2_152',
    'tf_inc_res_v2',
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
'tf_ens_adv_inc_res_v2'
    ]
if __name__=='__main__':
    for surrogate in list_nets:
        file = f'{surrogate}_loss.csv'
        include_methods = ['MIFGSM', 'IFGSM', 'MDI2FGSM', 'TranslationInvariantAttack', 'DI2FGSM',
                           'TI_DI_AITM', 'AMDI2FGSM', 'SI_NI_FGSM', 'EMI_FGSM', 'PIFGSM', 'admix_FGSM']
        df=pd.read_csv(file)
        df = df.drop(labels='Unnamed: 0', axis=1)
        def get_not_null_data(data, col):
            data = data[(data[col].notnull()) & (data[col] != "")]
            return data
        modified_file=pd.DataFrame(columns=include_methods)
        for method in include_methods:
            dem=get_not_null_data(df, method)
            dem.index = Series(range(1000))
            modified_file[method]=dem[method]
        modified_file.describe().to_csv(f"./loss/{surrogate}_loss.csv")
        # draw_pdf(modified_file,surrogate)
