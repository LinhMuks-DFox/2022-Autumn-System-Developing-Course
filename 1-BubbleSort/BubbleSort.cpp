#include <iostream>
#include <vector>
using std::cout;
using i32array_t = std::vector<int>;

class IntegerBubbleSorter {
public:
    inline void operator()(i32array_t& arr) const
    {
        this->sort(arr);
    }

private:
    void static inline swap(int& i, int& j)
    {
        auto t = i;
        i = j;
        j = t;
    }

    void static inline sort(i32array_t& arr)
    {
        int i, j, len = arr.size();
        for (i = 0; i < len - 1; i++)
            for (j = 0; j < len - 1 - i; j++)
                if (arr[j] > arr[j + 1])
                    swap(arr[j], arr[j + 1]);
    }
};
int main()
{
    i32array_t arr { 4, 5, 3, 6, 2, 5, 1 };
    auto print_arr = [&arr]() {for (const auto & i : arr) {cout << i << ", ";} cout << std::endl; };
    print_arr();
    IntegerBubbleSorter sorter;
    sorter(arr);
    print_arr();
    return 0;
}