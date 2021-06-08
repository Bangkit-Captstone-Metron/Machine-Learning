from imageio import imread
import pickle
import numpy as np
import os
import cv2

fake_path = 'training/fake/'
pristine_path = 'training/pristine/'
mask_path = fake_path + 'masks/'
numberOfBatch = 8

def count_255(mask):
    # print("Start counting")
    i=0
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row,col]==255:
                i+=1
    # print("Finish counting")
    return i


def sample_fake(img, mask):
    print("Start sampling")
    kernel_size = 64
    stride = 8
    threshold=1024

    samples = []

    total_samples = (img.shape[0] - kernel_size)//stride * (img.shape[1] - kernel_size)//stride
    print("Total samples: " + str(total_samples))
    # ind = 0
    for y_start in range(0, img.shape[0] - kernel_size + 1, stride):
        for x_start in range(0, img.shape[1] - kernel_size + 1, stride):
            
            # ind += 1
            # print("Sampling progress: " + str(ind/total_samples))

            c_255 = count_255(mask[y_start:y_start + kernel_size, x_start:x_start + kernel_size])

            # print("Appending to samples")
            if (c_255 > threshold) and (kernel_size * kernel_size - c_255 > threshold):
                samples.append(img[y_start:y_start + kernel_size, x_start:x_start + kernel_size, :3])

    return samples


def main():

    x_train_masks = []
    with open('pickle/images/x_train_masks_' + str(numberOfBatch) + '.pickle', 'rb') as f:
        x_train_masks.extend(pickle.load(f))

    with open('pickle/images_names/x_train_fakes_names.pickle', 'rb') as f:
        x_train_fakes_names = pickle.load(f)

    x_train_fakes_names_final = []
    x_train_fakes_names_final.extend(x_train_fakes_names[numberOfBatch*40:(numberOfBatch+1)*40])

    x_train_fake_images = []
    for img in x_train_fakes_names_final:
        x_train_fake_images.append(imread(fake_path + img))

    # Convert grayscale images to binary
    binaries=[]

    for grayscale in x_train_masks:
        blur = cv2.GaussianBlur(grayscale,(5,5),0)
        ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binaries.append(th)

    print("Panjang train fake images: " + str(len(x_train_fake_images)))
    print("Panjang binaries: " + str(len(binaries)))

    samples_fakes_arr = []

    j = 0
    for img, mask in zip(x_train_fake_images, binaries):
        print("Start iteration")
        samples=sample_fake(img, mask)
        print("Finish sampling with total number of samples " + str(len(samples)))
        for s in samples:
            samples_fakes_arr.append(s)
        j += 1
        print("Progress: " + str(j/len(x_train_fake_images)))
    
    samples_fakes_np = np.array(samples_fakes_arr)

    print('done')
    np.save('sample_images/k64 binary 25percent stride8/sample_fakes_'+str(numberOfBatch+1)+'.npy', samples_fakes_np)


if __name__ == '__main__':
    main()

