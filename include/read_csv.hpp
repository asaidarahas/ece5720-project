#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <fstream>

#pragma once

typedef unsigned char uchar;

namespace csv
{
    struct data
    {
        float a;
        float b;
        float c;
        float d;
        float e;
        std::string f;
    };

    struct Dataset
    {
        std::vector<data> train_feat; // training images
        std::vector<data> test_feat;  // test images
        std::vector<int> train_labels;   // training labels
        std::vector<int> test_labels;    // test labels
    };

    std::vector<data> read_feat(const std::string &path, std::size_t limit)
    {
        std::ifstream File(path.c_str());

        std::string line;

        std::vector<data> dataset{};

        if (File)
        {
            std::string header{};
            if (std::getline(File, header))
                std::cout << header << '\n';

            // Read all data
            data tmp{};
            char comma;
            int i = 0;
            while (File >> tmp.a >> comma >> tmp.b >> comma >> tmp.c >> comma >> tmp.d >> comma >> tmp.e >> comma >> tmp.f)
            {
                dataset.push_back(std::move(tmp));
                i++;
            }
            // Show everything
            for (const data &row : dataset)
                std::cout << row.a << '\t' << row.b << '\t' << row.c << '\t' << row.d << '\t' << row.e << '\t' << row.f << '\n';
        }
        else
        {
            throw std::runtime_error("Cannot open file `" + path + "`!");
        }
        return dataset;
    }

    std::vector<data> read_train_feat(const std::string &path, std::size_t limit)
    {
        return read_feat(path, limit);
    }

    std::vector<data> read_test_feat(const std::string &path, std::size_t limit)
    {
        return read_feat(path, limit);
    }

    Dataset read_dataset(const std::string &path, const std::string &filename, std::size_t training_limit = 0, std::size_t test_limit = 0)
    {
        Dataset dataset;

        dataset.train_feat = read_train_feat(path + '/Iris.csv', training_limit);

        dataset.test_feat = read_test_feat(path + '/Iris.csv', test_limit);

        return dataset;
    }

}
