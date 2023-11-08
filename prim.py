import cv2
import numpy as np

#Definimos una función para verificar si el camino entre dos vértices es válido, esta madre es el algoritmo de bresenham
def camino_valido(w, v, j, k, th2): # Recibimos la imagen como parametro (th2)
    # Función para comprobar si el camino entre dos vértices está despejado usando bresenham
    # En esta función, estamos verificando si el camino entre los vértices j y k
    # a lo largo de una línea recta está completamente despejado (todos los puntos muestreados son blancos).
    #w es el grafo que recibimos como parametro
    min_i = min(j, k) #verifica cual es el vertice con el indice menor
    max_i = max(j, k) #este verifica el vertice con el indice mayor
    x0, y0 = corner_coordinates[min_i]  # Obtenemos las coordenadas del vértice "min_i"
    x1, y1 = corner_coordinates[max_i] # Obtenemos las coordenadas del vértice "max_i"

    # Calculamos la diferencia en las coordenadas x e y entre los dos vértices
    dx = abs(x1 - x0) #calcula la longitud absoluta del segmento en x entre los vertices
    dy = -abs(y1 - y0) #calcula la longitud del segmento en x entre los vertices y se guarda como el negativo de su
                       #valor absoluto ya que le eje y esta invertido

    # Determinamos la dirección de incremento en x (sx) y y (sy) para avanzar a lo largo de la línea
    sx = 1 if x0 < x1 else -1 #aca checamos si el punto inicial en la coordenada x es menor al punto final entonces
                               #se establece en 1 y si no, en -1, esto indica hacia que direccion en x avanzaremos
    sy = 1 if y0 < y1 else -1 #esto hace lo mismo pero en el eje y
    # Inicializamos un error para rastrear el avance a lo largo de la línea

    err = dx + dy #la variable err es una varaible que nos determina el margen de error que podemos tener al trazar una
    #diagonal entre las coordenadas de los vertices al aplicar el algoritmo de bresenham

    while True:
        # Verificamos si el punto actual en la línea no es blanco (representa un obstáculo)
        if not all(th2[y0, x0] == [255, 255, 255]): #se esta comprobando si al menos uno de los canales de color del
                                            # píxel en la imagen th2 en las coordenadas (y0, x0) no es igual a blanco
            return False #retorna false si el pixel no es blanco

        if x0 == x1 and y0 == y1: #verificamos si las coordenadas actuales son iguales a las coordenadas destino, lo que
            #indicaria que ya llegamos al final de la linea
            break #si es asi nos salimos del bucle
        # Calculamos e2, que nos ayuda a determinar cuándo avanzar en x o en y
        e2 = 2 * err
        # Calcula el valor de e2, que es esencial para controlar el trazado de la línea
        # e2 se utiliza para tomar decisiones sobre cuándo avanzar en las coordenadas x o y
        # Se calcula como el doble del error acumulado hasta este punto (err)

        if e2 >= dy: #se evalua si el doble del error acumulado es mayor o igual a la diferencia de las coordenadas en y
            err += dy #si se cumple la condicion, entonces al error acumulado le sumamos dy
            x0 += sx #y a x0 le sumamos 1 o -1, o sea que avanzamos en el eje de las x

        if e2 <= dx: #si el doble del error acumulado es menor o igual a la diferencia de las coordenadas en x
            # entonces significa que nos debenos de mover en el eje de las y
            err += dx #para eso se incrementa el error acumulado en la direccion horizontal
            y0 += sy #h actualizamos la coordenada actual en y0

    # Si hemos recorrido toda la línea sin encontrar obstáculos, el camino se considera válido
    return True


# Definimos la función principal de Prim
def prim(w, n, s, th2):
    # w: La matriz de pesos que representa el grafo completo.
    # n: El número de vértices en el grafo.
    # s: El vértice de inicio desde el cual se construirá el árbol mínimo.
    # th2: La imagen que representa el mapa y se utiliza para verificar si el camino entre vértices es válido.

    # Inicializa un vector v para llevar un seguimiento de los vértices visitados.
    v = [0] * n #se inicializan todos los indices de la lista en 0
    v[s] = 1  # Marca el vértice de inicio como visitado (o sea el indice 0)

    # Inicializa una lista E (edge) para almacenar las aristas del árbol mínimo.
    E = []

    # Mientras la cantidad de aristas en E sea menor que (n - 1), continúa buscando aristas para agregar.
    while len(E) < n - 1:
        # Inicializa 'min_weight' con un valor infinito.
        min_weight = float('inf')
        # Variables para el vértice que se va a agregar y la arista.
        add_vertex = None
        edge = None
        # Itera a través de todos los vértices.
        for j in range(n):
            # Si el vértice j ya está en el árbol mínimo.
            if v[j] == 1:
                # Itera a través de todos los vértices no visitados.
                for k in range(n):
                    if v[k] == 0 and w[j][k] < min_weight: #evalua si k no ha sido visitado y
                                                           # verifica que le peso de j y k sea menor al peso minimo
                        # Verifica si el camino entre j y k es válido utilizando la función camino_valido
                        if camino_valido(w, v, j, k, th2):
                            # Actualiza la información de la arista a agregar.
                            add_vertex = k #si la funcion retorna true significa que el vertice destino k tiene
                                         # una arista valida con el vertice origen j
                            edge = (j, k)#la tupla edge ahora tiene los valores de j y k
                            min_weight = w[j][k] #el peso minimo se actualiza con el peso de la arista actual

        # Si se encontró una arista válida, marca el vértice como visitado y agrega la arista a 'E'.
        if edge is not None:
            v[add_vertex] = 1 #ahora el vertice que solia ser k pasa a estatus de visitado
            E.append(edge) # a la lista de aristas se le agrega la arista valida resultante
        else:
            # Si no se encuentra ninguna arista válida, retorna 'None'.
            return None

    # Una vez que se ha construido el árbol mínimo, devuelve la lista de aristas 'E'.
    return E


nombreMapa = "3"
mapa = cv2.imread('mapa' + nombreMapa + '.png')
vertices = np.load("verticeMapa" + nombreMapa + ".npy")

# Convertimos la imagen del mapa a escala de grises
gray = cv2.cvtColor(mapa, cv2.COLOR_BGR2GRAY)
# Definimos un kernel para operaciones de dilatación y erosión
kernel = np.ones((11, 11), np.uint8) #se define el keren 11x11 con enteros sin signo de 8 bits
# Aplicamos umbral binario a la imagen en escala de grises
ret, th1 = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY) #defino el umbral como 254, por lo que todos
# los pixeles con un valor mayor o igual, pasaran a ser 255, o sea blanco, y los menores pasan a ser negros, esto para
# binarizar la imagen por completo en blanco o negro, la imagen binarizada se guarda en th1 y el valor del umbral en ret

# Realizamos operaciones de dilatación y erosión en la imagen binaria
th1 = cv2.dilate(th1, kernel, 1) #se dilata la imagen th1 con el kernel establecido y se itera 1 vez
th1 = cv2.erode(th1, kernel, 1) #se erosiona la imagen th1 con el kernel establecido y se itera 1 vez
# Aplicamos un desenfoque Gaussiano a la imagen  usando un kernel de 5x5
th1 = cv2.GaussianBlur(th1, (5, 5), cv2.BORDER_DEFAULT)

# Aplicamos un segundo umbral binario a la imagen
ret, th2 = cv2.threshold(th1, 235, 255, cv2.THRESH_BINARY) #defino el umbral como 235, por lo que todos
# los pixeles con un valor mayor o igual, pasaran a ser 255, o sea blanco, y los menores pasan a ser negros, esto para
# binarizar la imagen por completo en blanco o negro, la imagen binarizada se guarda en th2

# Realizamos una operación de dilatación en la imagen binaria
th2 = cv2.dilate(th2, kernel, 1)
# Convertimos la imagen binaria a una imagen en escala de grises
th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)

# Obtenemos las coordenadas de los vértices a partir de los datos guardados
corner_coordinates = [(int(vertice[1]), int(vertice[0])) for vertice in vertices] #se crea una lista con tuplas que
# contiene las coordenadas x, y que se extraen de la variable vertices que almacena la informacion del archivo numpy

# Calculamos el número de vértices
num_vertices = len(corner_coordinates)

# Inicializamos una matriz para representar el grafo
graph = np.zeros((num_vertices, num_vertices))

# Calculamos las distancias entre los vértices, basicamente pitagoras
for i in range(num_vertices):
    for j in range(i + 1, num_vertices):
        # Obtenemos las coordenadas de los vértices i y j desde corner_coordinates
        vertice_i = corner_coordinates[i]
        vertice_j = corner_coordinates[j]

        # Calculamos la diferencia en coordenadas en el eje x e y
        coords_x = vertice_i[0] - vertice_j[0]
        coords_y = vertice_i[1] - vertice_j[1]

        # Calculamos la distancia usando el teorema de Pitágoras
        distancia = (coords_x ** 2 + coords_y ** 2) ** 0.5

        # Almacenamos la distancia en ambas direcciones en la matriz graph
        graph[i][j] = graph[j][i] = distancia

# Encontramos el árbol de expansión mínima utilizando el algoritmo de Prim
minimum_spanning_tree_edges = prim(graph, num_vertices, 0, th2)

# Verificamos si se encontró un árbol de expansión mínima
if minimum_spanning_tree_edges:
    # Dibujamos el árbol de expansión mínima en la imagen con líneas verdes
    for edge in minimum_spanning_tree_edges:
        u, v = edge #los valores de la tupla edge se almacenan en las variables u y v
        cv2.line(th2, corner_coordinates[u], corner_coordinates[v], (0,255,0), 2)
        # Dibujamos la linea entre los vertices u y v. Establecemos el color verde con un grosor de 2

    cv2.imshow('Arbol de Expansión Minima', th2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontro un arbol de expansion minima.")
