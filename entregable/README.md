# Explicación general del código

## Archivos
El código se divide en las carpetas `datos` y `modelo`. La primera está destinada a procesar los datos de la carpeta `datos\informacion` para generar estructuras utilizables. La segunda carpeta contiene el código directamente asociado al funcionamiento del modelo propuesto.

## Carpeta `datos`
La información que se procesa son los archivos Excel `abogados.xlsx`, `arbol.xlsx` y `casos.xlsx`. Todos estos fueron descargados de la carpeta compartida con Impacto Legal. El archivo `procesamiento.ipynb` carga los archivos anteriores y genera las estructuras necesarias para que el modelo funcione. La manera de procesar no es relevante. Lo que es importante son los archivos `pickle` que se crean. En particular:
* `abogados.pickle`: `DataFrame` con las columnas:
    - `id`: número entero que sirve como identificador del abogado.
    - `hb`: número de horas semanales disponibles del abogado.
    - `exp`: años de experiencia.
    - `areas`: lista de áreas de expertiz. Cada área se representa por un único número entero.
    - `declarados`: lista de servicios, subáreas, subsubáreas, etc., que han sido declarados por el abogado. Cada elemento se representa por un único número entero (codificación).
    - `realizados`: lista de servicios realizados en IL.
    - `cant`: lista tal que `cant[i]` es la cantidad de veces que se ha realizado el servicio `realizados[i]`.
    - `promedio`: lista tal que `promedio[i]` es la calificación promedio obtenida al realizar `realizados[i]`.
* `servicios.pickle`: `DataFrame` con las columnas:
    - `id`: número entero que sirve como identificador del servicio.
    - `promedio hs`: horas/semana que requiere el servicio en promedio.
    - `minimo hs`: horas/semana que requiere el servicio como mínimo.
    - `maximo hs`: horas/semana que requiere el servicio como máximo.
    - `promedio s`: semanas que requiere el servicio en promedio.
    - `minimo s`: semanas que requiere el servicio como mínimo.
    - `maximo s`: semanas que requiere el servicio como máximo.
* `casos.pickle`: lista de casos donde cada caso es una lista de los servicios que componen el caso. Los servicios están representados por el mismo identificador numérico de `servicios.pickle`.
* `padres.pickle`: lista donde `padres[i]` es el índice asociado al 'padre' del servicio, área, subárea u otro de índice `i`. El índice coincide con la codificación de los servicios, áreas, etc.

Para acceder a los nombres de los abogados y servicios, áreas, subáreas, etc., se tienen los diccionarios `decod_nombres.pickle` y `decodificacion.pickle`:

* `decod_nombres[i]` es el nombre del abogado de identificador `i`. 
* `decodificacion[i]` es el nombre del servicio, área, subárea u otro de índice `i`.

## Carpeta `modelo`
Esta carpeta almacena los archivos `instance.py`, `model.py`, `parameters.py` y `rating_function.py`.

### `instance.py`
Archivo que contiene la clase `Instance` que crea una *instancia* para el modelo. Se reciben obligatoriamente los siguientes parámetros:
* `cases`: estructura que se obtiene después de hacer *load* de `cases.pickle`.
* `services`: estructura que se obtiene después de hacer *load* de `services.pickle`.
* `lawyers`: estructura que se obtiene después de hacer *load* de `lawyers.picke`.
* `parents`: estructura que se obtiene después de hacer *load* de `parents.picke`.
* `tmin`: tiempo mínimo a asignar semanalmente a cualquier abogado.
* `base_cases`: lista de casos que llegan un día determinado. Cada caso se representa como un diccionario donde las *keys* son los índices de los servicios que componen el caso y los *values* son tuplas de la forma `(hweeks, weeks)` con `hweeks` la cantidad de horas/semana y `weeks` la cantidad de semanas que requiere cada servicio. Cada vez que se quiera determinar la asignación óptima para un caso o conjunto de casos es necesario ingresarlo en el formato que indica `base_cases`.

Hay otros parámetros que son opcionales y su uso depende de cuál valor toma el parámetro `mode`. Por defecto, `mode=saa`. En este caso, se deben proporcionar los siguientes parámetros para la simulación interna del método *Sample Average Aproximation* (SAA):
* `nscenarios`: número de escenarios a simular.
* `rate`: tasa de llegada de casos semanales.
* `hor`: horizonte de tiempo a considerar.
* `lambd`: factor de descuento temporal.
* `arrival`: número del día en el que llegan los `base_cases`. Es un número entero en el intervalo $[1,5]$. Por ejemplo, `arrival=1` significa que los casos llegan el lunes, `arrival=2` que llegan el martes y así sucesivamente.

En el archivo `parameters.py` hay valores prefijados para `nscenarios`, `rate`, `hor` y `lambd`. El otro modo posible es `mode=greedy`. Consiste en una instancia para un modelo puramente determinístico que no considera el futuro.

### `model.py`
Archivo que contiene la clase `ILModel`. Para inicializar la clase se recibe un objeto de clase `instance`. El funcionamiento de cada función está explicado en el mismo código de `model.py`.
La interfaz utilizada es `Pyomo` y el *solver* se puede fijar en la función `run`. Recomendamos el *solver Open Source* `glpk`.

### `rating_function.py`
Conjunto de funciones que modelan una función de *rating* de un abogado respecto a un servicio.

## Librerías
Las librerias externas que se utilizaron fueron `pyomo.environ`, `pyomo.opt`, `numpy`, `random`, `collections`, `math`, `pandas` y `pickle`.

