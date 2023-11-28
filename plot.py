import matplotlib.pyplot as plt
import os

def pad_green_red(ax, name):
    if "True" in name:
        ax.tick_params(color='green', labelcolor='green')
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
    else:
        ax.tick_params(color='red', labelcolor='red')
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
query_name = "csY5pYyTRIoJvLf-8d-i6g"
plt.figure(figsize=(10, 7), dpi=400)
plt.subplot(7,7,1)
img = plt.imread(f"vis_simclr/{query_name}.jpg")
plt.imshow(img, aspect="auto")
plt.axis('off')
plt.title("Query image", fontsize=7)
row = 0
f_name = ['SimCLR', "MoCov2", "BYOL", "SimSiam", "Barlow Twins", "VICReg"]
folder_name =  ['simclr', 'mocov2', 'byol', 'simsiam', 'bt', 'vicreg']
for j in range(len(folder_name)):
    ax = plt.subplot(7,7,row*7+2)
    ax.text(0.3, 0.5, f_name[j], fontsize=7)
    plt.axis('off')
    for i in range(1, 6):
        ax = plt.subplot(7,7,row*7+2+i)
        filename_single = [filename for filename in os.listdir(f"vis_{folder_name[j]}") if filename.startswith(f"{query_name}_{i-1}")][0]
        filename = f"vis_{folder_name[j]}/{filename_single}"
        img = plt.imread(filename)
        pad_green_red(
            ax, filename)
        plt.imshow(img, aspect="auto")
        plt.xticks([])
        plt.yticks([])
    row = row+1
plt.show()
plt.savefig("hello.png", bbox_inches='tight')
