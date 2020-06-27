import os
import re
from urllib.parse import unquote
import numpy as np


def incidence_mat_generate(data_dir, files, page_index):
    M = np.zeros((len(files), len(files)))

    for j, file in enumerate(files):
        with open(os.path.join(data_dir, file), 'r', encoding="utf-8") as f:
            text = f.read()

        links = re.findall('href="/wiki/(.+?)"', text)
        links = [unquote(i) for i in links]

        for l in links:
            if l not in page_index:
                continue
            i = page_index[l]
            M[i][j] = 1

    return np.nan_to_num(M / M.sum(0))


def pagerank(M, num_iterations=100, d=0.85):
    N = M.shape[1]
    v = np.random.rand(N, 1)
    v = v / np.linalg.norm(v, 1)
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = M_hat @ v
    return v


def hits(A):
    AAt = np.dot(A, A.T)
    AtA = np.dot(A.T, A)

    h_eig_value, h = np.linalg.eig(AAt)
    h_eig_value_argmax = np.argmax(h_eig_value)
    h = np.real(h[:,h_eig_value_argmax]).astype(float)


    a_eig_value, a = np.linalg.eig(AtA)
    a_eig_value_argmax = np.argmax(a_eig_value)
    a = np.real(a[:,a_eig_value_argmax]).astype(float)

    return a, h


if __name__== "__main__":
    
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./data/raw")
    files = os.listdir(data_dir)

    pages = [re.sub(f'\.html$', '', p) for p in files]
    page_index = {p: i for i, p in enumerate(pages)}
    M = incidence_mat_generate(data_dir, files, page_index)

    rank_pages = pagerank(M, d=0.89)

    sort_buf = np.array([rank_pages.reshape(-1), np.arange(rank_pages.shape[0])]).T
    sort_buf = sort_buf[sort_buf[:,0].argsort()]
    idx = sort_buf[:,1][::-1].astype(int)

    print("[pagerank: all]")
    for i in range(10):
        print(files[idx[i]])

    print("\n")

    search_idx = [page_index[k] for k in page_index if "Армстронг" in k]
    search_rank = [rank_pages[s] for s in search_idx]

    sort_buf = np.array([np.array(search_rank).reshape(-1), np.array(search_idx).reshape(-1)]).T
    sort_buf = sort_buf[sort_buf[:,0].argsort()]
    idx = sort_buf[:,1][::-1].astype(int)

    print(u"[pagerank: Армстронг]")
    for i in range(10):
        print(files[idx[i]])


    print("\n")


    A = np.zeros(M.shape)
    A[search_idx] = M[search_idx]
    A[:,search_idx] = M[:,search_idx]

    authority, hub = hits(A)
    authority = authority[search_idx]
    hub = hub[search_idx]
    
    sort_buf = np.array([authority, search_idx]).T
    sort_buf = sort_buf[sort_buf[:,0].argsort()]
    idx_authority_sort = sort_buf[:,1][::-1].astype(int)

    print(u"[hits: authority: Армстронг]")
    for i in range(10):
        print(files[idx_authority_sort[i]])

    print("\n")

    sort_buf = np.array([hub, search_idx]).T
    sort_buf = sort_buf[sort_buf[:,0].argsort()]
    idx_hub_sort = sort_buf[:,1][::-1].astype(int)
    
    print(u"[hits: hub: Армстронг]")
    for i in range(10):
        print(files[idx_hub_sort[i]])