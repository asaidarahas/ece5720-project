#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#pragma once

typedef unsigned char uchar;

namespace mnist
{
    struct MnistDataset
    {
        // std::vector<std::vector<unsigned char>> training_images;
        uchar **training_images; // training images
        uchar **test_images;     // test images
        uchar *training_labels;  // training labels
        uchar *test_labels;      // test labels
    };

    int reverseInt(int i)
    {
        uchar c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

    uchar **read_mnist_images(const std::string &path, std::size_t limit)
    {
        std::ifstream file(path, std::ios::binary);

        if (file.is_open())
        {
            int magic_number;
            int number_of_images;
            int n_rows;
            int n_cols;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            file.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);

            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);

            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);

            uchar **dataset = new uchar *[number_of_images];
            for (int i = 0; i < number_of_images; ++i)
            {
                dataset[i] = new uchar[n_rows * n_cols];
                file.read((char *)dataset[i], n_rows * n_cols);
            }
            return dataset;
        }
        else
        {
            throw std::runtime_error("Cannot open file `" + path + "`!");
        }
    }

    uchar *read_mnist_labels(const std::string &path, std::size_t limit)
    {
        std::ifstream file(path, std::ios::binary);

        if (file.is_open())
        {
            int magic_number;
            int number_of_labels;
            int n_rows;
            int n_cols;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            file.read((char *)&number_of_labels, sizeof(number_of_labels));
            number_of_labels = reverseInt(number_of_labels);

            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);

            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);

            uchar *dataset = new uchar[number_of_labels];
            for (int i = 0; i < number_of_labels; ++i)
            {
                file.read((char *)&dataset[i], 1);
            }
            return dataset;
        }
        else
        {
            throw std::runtime_error("Cannot open file `" + path + "`!");
        }
    }

    uchar **read_training_images(const std::string &path, std::size_t limit)
    {
        return read_mnist_images(path + "/train-images-idx3-ubyte", limit);
    }

    uchar **read_test_images(const std::string &path, std::size_t limit)
    {
        return read_mnist_images(path + "/t10k-images-idx3-ubyte", limit);
    }

    uchar *read_training_labels(const std::string &path, std::size_t limit)
    {
        return read_mnist_labels(path + "/train-labels-idx1-ubyte", limit);
    }

    uchar *read_test_labels(const std::string &path, std::size_t limit)
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

} // end of namespace mnist