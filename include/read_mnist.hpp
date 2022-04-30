#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdexcept>

#pragma once

typedef unsigned char uchar;

namespace mnist
{
    struct MnistDataset
    {
        std::vector<uint8_t> training_images; // training images
        std::vector<uint8_t> test_images;     // test images
        std::vector<uint8_t> training_labels; // training labels
        std::vector<uint8_t> test_labels;     // test labels
    };

    inline int reverseInt(int i)
    {
        uchar c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

    void get_image_size(const std::string &path, size_t &width, size_t &height)
    {
        std::string full_path = path + "/train-images-idx3-ubyte";
        FILE *fp = fopen(full_path.c_str(), "rb");

        if (fp)
        {
            uint32_t magic_number;
            uint32_t number_of_images;

            fread(&magic_number, sizeof(uint32_t), 1, fp);
            magic_number = reverseInt(magic_number);

            fread(&number_of_images, sizeof(uint32_t), 1, fp);
            number_of_images = reverseInt(number_of_images);

            fread(&width, sizeof(int), 1, fp);
            width = reverseInt(width);

            fread(&height, sizeof(int), 1, fp);
            height = reverseInt(height);
        }
        else
        {
            throw std::runtime_error("Cannot open file `" + path + "`!");
        }
        fclose(fp);
    }

    std::vector<uint8_t> read_mnist_images(const std::string &path, std::size_t limit)
    {
        FILE *fp = fopen(path.c_str(), "rb");

        if (fp)
        {
            uint32_t magic_number;
            uint32_t number_of_images;
            uint32_t width;
            uint32_t height;

            fread(&magic_number, sizeof(uint32_t), 1, fp);
            magic_number = reverseInt(magic_number);

            fread(&number_of_images, sizeof(uint32_t), 1, fp);
            number_of_images = reverseInt(number_of_images);

            fread(&width, sizeof(uint32_t), 1, fp);
            width = reverseInt(width);

            fread(&height, sizeof(uint32_t), 1, fp);
            height = reverseInt(height);

            std::vector<uint8_t> dataset(number_of_images * width * height);
            fread(&dataset[0], sizeof(uint8_t), number_of_images * width * height, fp);

            fclose(fp);
            return dataset;
        }
        else
        {
            throw std::runtime_error("Cannot open file `" + path + "`!");
        }
        fclose(fp);
    }

    std::vector<uint8_t> read_mnist_labels(const std::string &path, std::size_t limit)
    {
        FILE *fp = fopen(path.c_str(), "rb");

        if (fp)
        {
            uint32_t magic_number;
            uint32_t number_of_labels;

            fread(&magic_number, sizeof(uint32_t), 1, fp);
            magic_number = reverseInt(magic_number);

            fread(&number_of_labels, sizeof(uint32_t), 1, fp);
            number_of_labels = reverseInt(number_of_labels);

            std::vector<uint8_t> dataset(number_of_labels);
            fread(&dataset[0], sizeof(uint8_t), number_of_labels, fp);

            fclose(fp);
            return dataset;
        }
        else
        {
            throw std::runtime_error("Cannot open file `" + path + "`!");
        }
        fclose(fp);
    }

    std::vector<uint8_t> read_training_images(const std::string &path, std::size_t limit)
    {
        return read_mnist_images(path + "/train-images-idx3-ubyte", limit);
    }

    std::vector<uint8_t> read_test_images(const std::string &path, std::size_t limit)
    {
        return read_mnist_images(path + "/t10k-images-idx3-ubyte", limit);
    }

    std::vector<uint8_t> read_training_labels(const std::string &path, std::size_t limit)
    {
        return read_mnist_labels(path + "/train-labels-idx1-ubyte", limit);
    }

    std::vector<uint8_t> read_test_labels(const std::string &path, std::size_t limit)
    {
        return read_mnist_labels(path + "/t10k-labels-idx1-ubyte", limit);
    }

    MnistDataset read_dataset(const std::string &path, std::size_t training_limit = 0, std::size_t test_limit = 0)
    {
        MnistDataset dataset;

        dataset.training_images = read_training_images(path, training_limit);
        dataset.training_labels = read_training_labels(path, training_limit);

        dataset.test_images = read_test_images(path, test_limit);
        dataset.test_labels = read_test_labels(path, test_limit);

        return dataset;
    }

}
