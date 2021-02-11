import numpy as np


def best_match(known_faces: np.array, face: np.array, tolerance: float) -> tuple:
    """Funcion que mide la distancia euclidea entre un vector y las componente de una matriz para
    estimar la semejanza entre ambos y compara con un rango de tolerancia para aceptar si es el
    mismo vector o no
    :param known_faces np.array contiene la matriz de embeddings
    :param face np.array contiene la codificacion de la cara comparar
    :param tolerance float es el umbral de decisión que determinara si coincide o no.
    :returns idx, d_e, match tuple tupla que contiene indice, distancia euclidea e indicacion de match (True para match)
    """
    assert (isinstance(face, np.ndarray) and (isinstance(known_faces, np.ndarray))), f"opera unicamente arrays de numpy"
    assert (known_faces.shape[1] == face.shape[0]), f"Las dimension de la cara suministrada es incorrecta"
    d_e = np.linalg.norm(known_faces-face, axis=1)  # vector
    idx = np.argmin(d_e)  # indice
    match = False
    if d_e[idx] <= tolerance:
        match = True
    return idx, d_e[idx], match


if __name__ == "__main__":
    np.random.seed(1)
    known_ids = np.random.rand(100, 128).astype(np.float32)
    face_encoding = np.random.rand(128).astype(np.float32)
    print(best_match(known_faces=known_ids, face=face_encoding, tolerance=4.1))
