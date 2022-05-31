import http
import re
import socket
import urllib.request
from urllib.error import HTTPError

import numpy as np


def findHTTP(root: str):
    urls = []
    skips = ['whatsapp', 'mailto', '.gif', '.jpg', '.jpeg', '.png',
             '.pdf', '.css', '.asp', '.mwc', '.ram', '.cgi',
             'lmscadsi', 'cybernet', 'w3.org', 'google', 'yahoo',
             'scripts', 'netscape', 'shockwave', 'webex', 'fansonly']
    not_skips = ['http://', 'https://']

    # Try to open a page.
    try:
        with urllib.request.urlopen(root, timeout=5) as response:
            # Read and decode html
            html = response.read()
            plaintext = html.decode('utf8', "ignore")

            # Follow the links from the open page
            # A link starts with 'http:' and ends with the next double quote.
            links = re.findall("href=[\"\'](.*?)[\"\']", plaintext)
            links += (re.findall("HREF=[\"\'](.*?)[\"\']", plaintext))
            for link in links:
                # Remove last '/' and strip the single link
                if len(link) > 1 and link[-1] == '/':
                    link = str(link[:-1])
                link.strip()

                # Look for links that should not be skipped.
                if not any(word in link for word in skips) and any(word in link for word in not_skips):
                    urls.append(link)

    except (HTTPError, urllib.error.URLError, ConnectionResetError, http.client.InvalidURL) as err:
        print(root, " ERRORE: ", err)

    except socket.timeout:
        print("connection's timeout expired")

    return urls


def surfer(root, n, verbose=False):
    """ 
    Create the adjacency graph of a portion of the Web.
    [U,G] = surfer(root,n) starts at the URL root and follows
    Web links until it forms an adjacency graph with n nodes.

    :param root: root url
    :param n: urls to surf
    :param verbose: if True, print information
    :return: U = a cell array of n strings, the URLs of the nodes.
    :return: G = an n-by-n sparse matrix with G(i,j)=1 if node j is linked to node i.
    :rtype: list, np.array
    """

    # Initialize
    U = []
    G = np.zeros((n, n))
    hashURL = []
    hashURL.append(hash(root))
    U.append(root)

    next_root_idx = 0
    j = 0

    for j in range(n):

        current_root = U[j]

        # Find HTTP list.
        pages = findHTTP(current_root)
        print("Page #", j, "=", U[j])
        if verbose:
            print(len(pages), "link found")

        for page in pages:
            hash_page = hash(page)
            try:
                # Check if page is already in url list.
                i = hashURL.index(hash_page)
                # Add a new link in G.
                if i != j and U[i] == page:
                    G[i, j] = 1
            except ValueError:
                # Add a new url to the graph there if are fewer than n
                if next_root_idx < n-1:
                    next_root_idx += 1
                    hashURL.append(hash_page)
                    U.append(page)
                    G[next_root_idx, j] = 1

    return U, G


def pagerank(G: np.array, p: float, verbose=False):
    """
    Calcola matrice dei coefficienti A e matrice dei termini noti b per il calcolo del pageRank delle pagine individuate da G

    :param G: matrice di connettività
    :param p: numero reale compreso tra 0 e 1
    :param verbose: se True stampa informazioni su A e b
    :return: matrice dei coefficienti A e matrice dei termini noti b
    :rtype: np.array, np.array
    """

    n = np.shape(G)[0]  # numero di pagine = numero di righe o colonne di G

    # Modifica di G
    G = G-np.diag(np.diag(G))  # Eliminazione link autoreferenziali
    for i in range(n):  # Riempimento colonne nulle
        if sum(G[:, i]) == 0:
            G[:, i] = 1
            G[i, i] = 0

    # Costruzione b
    delta = (1-p)/n  # Importanza minima
    e = np.ones((n, 1))  # Vettore colonna di tutti 1
    b = delta * e
    if verbose:
        print("b =")
        print(b)

    # costruzione A e Ap
    I = np.eye(n)  # matrice identità nXn
    c = sum(G)  # vettore riga somma degli elementi della colonna
    cInv = 1./c  # operazione componente per componente
    D = np.diag(cInv)
    A = I - p*(np.dot(G, D))
    Ap = p * np.dot(G, D) + delta * np.dot(e, e.T)

    if verbose:
        print("D =")
        print(D)
        print("A =")
        print(A)
        print("Ap =")
        print(Ap)

    return A, Ap, b
