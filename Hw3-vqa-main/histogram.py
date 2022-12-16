import torch
from matplotlib import pyplot as plt

def Convert(tup, di):
    di = dict(tup)
    return di

def countFreq(arr, n):
    mp = dict()

    # Traverse through array elements
    # and count frequencies

    for i in range(n):
        if arr[i] in mp.keys():
            mp[arr[i]] += 1
        else:
            mp[arr[i]] = 1

    # Traverse through map and print
    # frequencies

    return mp


if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    out = torch.load('output.pt')
    print(out.size())
    out = out.tolist()
    mp = countFreq(out, len(out))
    mp_sorted = sorted(mp.items(), key=lambda x: x[1], reverse=True)
    print(mp_sorted)

    mp_sorted_dict = {}
    mp_sorted_dict = Convert(mp_sorted, mp_sorted_dict)
    answer = list(mp_sorted_dict.keys())
    freq = list(mp_sorted_dict.values())
    print(answer)
    print(freq)
    plt.bar(answer, freq)
    plt.title("Simple Model histogram")
    plt.ylabel("Frequencies")
    # plt.xticks(range(len(answer)), answer)
    plt.savefig("Simple-model-hist.png")
