from src.arc_dslearn.arc_dsl.arc_types import *


def identity(x: Any) -> Any:
    """Identity function"""
    return x


def add(a: Numerical, b: Numerical) -> Numerical:
    """Addition"""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + b[0], a[1] + b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a + b[0], a + b[1])
    return (a[0] + b, a[1] + b)


def subtract(a: Numerical, b: Numerical) -> Numerical:
    """Subtraction"""
    if isinstance(a, int) and isinstance(b, int):
        return a - b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] - b[0], a[1] - b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a - b[0], a - b[1])
    return (a[0] - b, a[1] - b)


def multiply(a: Numerical, b: Numerical) -> Numerical:
    """Multiplication"""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] * b[0], a[1] * b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a * b[0], a * b[1])
    return (a[0] * b, a[1] * b)


def divide(a: Numerical, b: Numerical) -> Numerical:
    """Floor division"""
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    elif isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] // b[0], a[1] // b[1])
    elif isinstance(a, int) and isinstance(b, tuple):
        return (a // b[0], a // b[1])
    return (a[0] // b, a[1] // b)


def invert(n: Numerical) -> Numerical:
    """Inversion with respect to addition"""
    return -n if isinstance(n, int) else (-n[0], -n[1])


def even(n: Integer) -> Boolean:
    """Evenness"""
    return n % 2 == 0


def double(n: Numerical) -> Numerical:
    """Scaling by two"""
    return n * 2 if isinstance(n, int) else (n[0] * 2, n[1] * 2)


def halve(n: Numerical) -> Numerical:
    """Scaling by one half"""
    return n // 2 if isinstance(n, int) else (n[0] // 2, n[1] // 2)


def flip(b: Boolean) -> Boolean:
    """Logical not"""
    return not b


def equality(a: Any, b: Any) -> Boolean:
    """Equality"""
    return a == b


def contained(value: Any, container: Container) -> Boolean:
    """Element of"""
    return value in container


def combine(a: Container, b: Container) -> Container:
    """Union"""
    return type(a)((*a, *b))


def intersection(a: FrozenSet, b: FrozenSet) -> FrozenSet:
    """Returns the intersection of two containers"""
    return a & b


def difference(a: FrozenSet, b: FrozenSet) -> FrozenSet:
    """Set difference"""
    return type(a)(e for e in a if e not in b)


def dedupe(tup: Tuple) -> Tuple:
    """Remove duplicates"""
    return tuple(e for i, e in enumerate(tup) if tup.index(e) == i)


def order(container: Container, compfunc: Callable) -> Tuple:
    """Order container by custom key"""
    return tuple(sorted(container, key=compfunc))


def repeat(item: Any, num: Integer) -> Tuple:
    """Repetition of item within vector"""
    return tuple(item for i in range(num))


def greater(a: Integer, b: Integer) -> Boolean:
    """Greater"""
    return a > b


def size(container: Container) -> Integer:
    """Cardinality"""
    return len(container)


def merge(containers: ContainerContainer) -> Container:
    """Merging"""
    return type(containers)(e for c in containers for e in c)


def maximum(container: IntegerSet) -> Integer:
    """Maximum"""
    return max(container, default=0)


def minimum(container: IntegerSet) -> Integer:
    """Minimum"""
    return min(container, default=0)


def valmax(container: Container, compfunc: Callable) -> Integer:
    """Maximum by custom function"""
    return compfunc(max(container, key=compfunc, default=0))


def valmin(container: Container, compfunc: Callable) -> Integer:
    """Minimum by custom function"""
    return compfunc(min(container, key=compfunc, default=0))


def argmax(container: Container, compfunc: Callable) -> Any:
    """Largest item by custom order"""
    return max(container, key=compfunc)


def argmin(container: Container, compfunc: Callable) -> Any:
    """Smallest item by custom order"""
    return min(container, key=compfunc)


def mostcommon(container: Container) -> Any:
    """Most common item"""
    return max(set(container), key=container.count)


def leastcommon(container: Container) -> Any:
    """Least common item"""
    return min(set(container), key=container.count)


def initset(value: Any) -> FrozenSet:
    """Initialize container"""
    return frozenset({value})


def both(a: Boolean, b: Boolean) -> Boolean:
    """Logical and"""
    return a and b


def either(a: Boolean, b: Boolean) -> Boolean:
    """Logical or"""
    return a or b


def increment(x: Numerical) -> Numerical:
    """Incrementing"""
    return x + 1 if isinstance(x, int) else (x[0] + 1, x[1] + 1)


def decrement(x: Numerical) -> Numerical:
    """Decrementing"""
    return x - 1 if isinstance(x, int) else (x[0] - 1, x[1] - 1)


def crement(x: Numerical) -> Numerical:
    """Incrementing positive and decrementing negative"""
    if isinstance(x, int):
        return 0 if x == 0 else (x + 1 if x > 0 else x - 1)
    return (
        0 if x[0] == 0 else (x[0] + 1 if x[0] > 0 else x[0] - 1),
        0 if x[1] == 0 else (x[1] + 1 if x[1] > 0 else x[1] - 1),
    )


def sign(x: Numerical) -> Numerical:
    """Sign"""
    if isinstance(x, int):
        return 0 if x == 0 else (1 if x > 0 else -1)
    return (
        0 if x[0] == 0 else (1 if x[0] > 0 else -1),
        0 if x[1] == 0 else (1 if x[1] > 0 else -1),
    )


def positive(x: Integer) -> Boolean:
    """Positive"""
    return x > 0


def toivec(i: Integer) -> IntegerTuple:
    """Vector pointing vertically"""
    return (i, 0)


def tojvec(j: Integer) -> IntegerTuple:
    """Vector pointing horizontally"""
    return (0, j)


def sfilter(container: Container, condition: Callable) -> Container:
    """Keep elements in container that satisfy condition"""
    return type(container)(e for e in container if condition(e))


def mfilter(container: Container, function: Callable) -> FrozenSet:
    """Filter and merge"""
    return merge(sfilter(container, function))


def extract(container: Container, condition: Callable) -> Any:
    """First element of container that satisfies condition"""
    return next(e for e in container if condition(e))


def totuple(container: FrozenSet) -> Tuple:
    """Conversion to tuple"""
    return tuple(container)


def first(container: Container) -> Any:
    """First item of container"""
    return next(iter(container))


def last(container: Container) -> Any:
    """Last item of container"""
    return max(enumerate(container))[1]


def insert(value: Any, container: FrozenSet) -> FrozenSet:
    """Insert item into container"""
    return container.union(frozenset({value}))


def remove(value: Any, container: Container) -> Container:
    """Remove item from container"""
    return type(container)(e for e in container if e != value)


def other(container: Container, value: Any) -> Any:
    """Other value in the container"""
    return first(remove(value, container))


def interval(start: Integer, stop: Integer, step: Integer) -> Tuple:
    """Range"""
    return tuple(range(start, stop, step))


def astuple(a: Integer, b: Integer) -> IntegerTuple:
    """Constructs a tuple"""
    return (a, b)


def product(a: Container, b: Container) -> FrozenSet:
    """Cartesian product"""
    return frozenset((i, j) for j in b for i in a)


def pair(a: Tuple, b: Tuple) -> TupleTuple:
    """Zipping of two tuples"""
    return tuple(zip(a, b, strict=False))


def branch(condition: Boolean, a: Any, b: Any) -> Any:
    """If else branching"""
    return a if condition else b


def compose(outer: Callable, inner: Callable) -> Callable:
    """Function composition"""
    return lambda x: outer(inner(x))


def chain(
    h: Callable,
    g: Callable,
    f: Callable,
) -> Callable:
    """Function composition with three functions"""
    return lambda x: h(g(f(x)))


def matcher(function: Callable, target: Any) -> Callable:
    """Construction of equality function"""
    return lambda x: function(x) == target


def rbind(function: Callable, fixed: Any) -> Callable:
    """Fix the rightmost argument"""
    n = function.__code__.co_argcount
    if n == 2:
        return lambda x: function(x, fixed)
    elif n == 3:
        return lambda x, y: function(x, y, fixed)
    else:
        return lambda x, y, z: function(x, y, z, fixed)


def lbind(function: Callable, fixed: Any) -> Callable:
    """Fix the leftmost argument"""
    n = function.__code__.co_argcount
    if n == 2:
        return lambda y: function(fixed, y)
    elif n == 3:
        return lambda y, z: function(fixed, y, z)
    else:
        return lambda y, z, a: function(fixed, y, z, a)


def power(function: Callable, n: Integer) -> Callable:
    """Power of function"""
    if n == 1:
        return function
    return compose(function, power(function, n - 1))


def fork(outer: Callable, a: Callable, b: Callable) -> Callable:
    """Creates a wrapper function"""
    return lambda x: outer(a(x), b(x))


def apply(function: Callable, container: Container) -> Container:
    """Apply function to each item in container"""
    return type(container)(function(e) for e in container)


def rapply(functions: Container, value: Any) -> Container:
    """Apply each function in container to value"""
    return type(functions)(function(value) for function in functions)


def mapply(function: Callable, container: ContainerContainer) -> FrozenSet:
    """Apply and merge"""
    return merge(apply(function, container))


def papply(function: Callable, a: Tuple, b: Tuple) -> Tuple:
    """Apply function on two vectors"""
    return tuple(function(i, j) for i, j in zip(a, b, strict=False))


def mpapply(function: Callable, a: Tuple, b: Tuple) -> Tuple:
    """Apply function on two vectors and merge"""
    return merge(papply(function, a, b))


def prapply(function: Callable, a: Container, b: Container) -> FrozenSet:
    """Apply function on cartesian product"""
    return frozenset(function(i, j) for j in b for i in a)


def mostcolor(element: Element) -> Integer:
    """Most common color"""
    values = (
        [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    )
    return max(set(values), key=values.count)


def leastcolor(element: Element) -> Integer:
    """Least common color"""
    values = (
        [v for r in element for v in r] if isinstance(element, tuple) else [v for v, _ in element]
    )
    return min(set(values), key=values.count)


def height(piece: Piece) -> Integer:
    """Height of grid or patch"""
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece)
    return lowermost(piece) - uppermost(piece) + 1


def width(piece: Piece) -> Integer:
    """Width of grid or patch"""
    if len(piece) == 0:
        return 0
    if isinstance(piece, tuple):
        return len(piece[0])
    return rightmost(piece) - leftmost(piece) + 1


def shape(piece: Piece) -> IntegerTuple:
    """Height and width of grid or patch"""
    return (height(piece), width(piece))


def portrait(piece: Piece) -> Boolean:
    """Whether height is greater than width"""
    return height(piece) > width(piece)


def colorcount(element: Element, value: Integer) -> Integer:
    """Number of cells with color"""
    if isinstance(element, tuple):
        return sum(row.count(value) for row in element)
    return sum(v == value for v, _ in element)


def colorfilter(objs: Objects, value: Integer) -> Objects:
    """Filter objects by color"""
    return frozenset(obj for obj in objs if next(iter(obj))[0] == value)


def sizefilter(container: Container, n: Integer) -> FrozenSet:
    """Filter items by size"""
    return frozenset(item for item in container if len(item) == n)


def asindices(grid: Grid) -> Indices:
    """Indices of all grid cells"""
    return frozenset((i, j) for i in range(len(grid)) for j in range(len(grid[0])))


def ofcolor(grid: Grid, value: Integer) -> Indices:
    """Indices of all grid cells with value"""
    return frozenset((i, j) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)


def ulcorner(patch: Patch) -> IntegerTuple:
    """Index of upper left corner"""
    return tuple(map(min, zip(*toindices(patch), strict=False)))


def urcorner(patch: Patch) -> IntegerTuple:
    """Index of upper right corner"""
    return tuple(
        map(
            lambda ix: {0: min, 1: max}[ix[0]](ix[1]),
            enumerate(zip(*toindices(patch), strict=False)),
        )
    )


def llcorner(patch: Patch) -> IntegerTuple:
    """Index of lower left corner"""
    return tuple(
        map(
            lambda ix: {0: max, 1: min}[ix[0]](ix[1]),
            enumerate(zip(*toindices(patch), strict=False)),
        )
    )


def lrcorner(patch: Patch) -> IntegerTuple:
    """Index of lower right corner"""
    return tuple(map(max, zip(*toindices(patch), strict=False)))


def crop(grid: Grid, start: IntegerTuple, dims: IntegerTuple) -> Grid:
    """Subgrid specified by start and dimension"""
    return tuple(r[start[1] : start[1] + dims[1]] for r in grid[start[0] : start[0] + dims[0]])


def toindices(patch: Patch) -> Indices:
    """Indices of object cells"""
    if len(patch) == 0:
        return frozenset()
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset(index for value, index in patch)
    return patch


def recolor(value: Integer, patch: Patch) -> Object:
    """Recolor patch"""
    return frozenset((value, index) for index in toindices(patch))


def shift(patch: Patch, directions: IntegerTuple) -> Patch:
    """Shift patch"""
    if len(patch) == 0:
        return patch
    di, dj = directions
    if isinstance(next(iter(patch))[1], tuple):
        return frozenset((value, (i + di, j + dj)) for value, (i, j) in patch)
    return frozenset((i + di, j + dj) for i, j in patch)


def normalize(patch: Patch) -> Patch:
    """Moves upper left corner to origin"""
    if len(patch) == 0:
        return patch
    return shift(patch, (-uppermost(patch), -leftmost(patch)))


def dneighbors(loc: IntegerTuple) -> Indices:
    """Directly adjacent indices"""
    return frozenset({
        (loc[0] - 1, loc[1]),
        (loc[0] + 1, loc[1]),
        (loc[0], loc[1] - 1),
        (loc[0], loc[1] + 1),
    })


def ineighbors(loc: IntegerTuple) -> Indices:
    """Diagonally adjacent indices"""
    return frozenset({
        (loc[0] - 1, loc[1] - 1),
        (loc[0] - 1, loc[1] + 1),
        (loc[0] + 1, loc[1] - 1),
        (loc[0] + 1, loc[1] + 1),
    })


def neighbors(loc: IntegerTuple) -> Indices:
    """Adjacent indices"""
    return dneighbors(loc) | ineighbors(loc)


def objects(grid: Grid, univalued: Boolean, diagonal: Boolean, without_bg: Boolean) -> Objects:
    """Objects occurring on the grid"""
    bg = mostcolor(grid) if without_bg else None
    objs = set()
    occupied = set()
    h, w = len(grid), len(grid[0])
    unvisited = asindices(grid)
    diagfun = neighbors if diagonal else dneighbors
    for loc in unvisited:
        if loc in occupied:
            continue
        val = grid[loc[0]][loc[1]]
        if val == bg:
            continue
        obj = {(val, loc)}
        cands = {loc}
        while len(cands) > 0:
            neighborhood = set()
            for cand in cands:
                v = grid[cand[0]][cand[1]]
                if (val == v) if univalued else (v != bg):
                    obj.add((v, cand))
                    occupied.add(cand)
                    neighborhood |= {(i, j) for i, j in diagfun(cand) if 0 <= i < h and 0 <= j < w}
            cands = neighborhood - occupied
        objs.add(frozenset(obj))
    return frozenset(objs)


def partition(grid: Grid) -> Objects:
    """Each cell with the same value part of the same object"""
    return frozenset(
        frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)
        for value in palette(grid)
    )


def fgpartition(grid: Grid) -> Objects:
    """Each cell with the same value part of the same object without background"""
    return frozenset(
        frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r) if v == value)
        for value in palette(grid) - {mostcolor(grid)}
    )


def uppermost(patch: Patch) -> Integer:
    """Row index of uppermost occupied cell"""
    return min(i for i, j in toindices(patch))


def lowermost(patch: Patch) -> Integer:
    """Row index of lowermost occupied cell"""
    return max(i for i, j in toindices(patch))


def leftmost(patch: Patch) -> Integer:
    """Column index of leftmost occupied cell"""
    return min(j for i, j in toindices(patch))


def rightmost(patch: Patch) -> Integer:
    """Column index of rightmost occupied cell"""
    return max(j for i, j in toindices(patch))


def square(piece: Piece) -> Boolean:
    """Whether the piece forms a square"""
    return (
        len(piece) == len(piece[0])
        if isinstance(piece, tuple)
        else height(piece) * width(piece) == len(piece) and height(piece) == width(piece)
    )


def vline(patch: Patch) -> Boolean:
    """Whether the piece forms a vertical line"""
    return height(patch) == len(patch) and width(patch) == 1


def hline(patch: Patch) -> Boolean:
    """Whether the piece forms a horizontal line"""
    return width(patch) == len(patch) and height(patch) == 1


def hmatching(a: Patch, b: Patch) -> Boolean:
    """Whether there exists a row for which both patches have cells"""
    return len(set(i for i, j in toindices(a)) & set(i for i, j in toindices(b))) > 0


def vmatching(a: Patch, b: Patch) -> Boolean:
    """Whether there exists a column for which both patches have cells"""
    return len(set(j for i, j in toindices(a)) & set(j for i, j in toindices(b))) > 0


def manhattan(a: Patch, b: Patch) -> Integer:
    """Closest manhattan distance between two patches"""
    return min(abs(ai - bi) + abs(aj - bj) for ai, aj in toindices(a) for bi, bj in toindices(b))


def adjacent(a: Patch, b: Patch) -> Boolean:
    """Whether two patches are adjacent"""
    return manhattan(a, b) == 1


def bordering(patch: Patch, grid: Grid) -> Boolean:
    """Whether a patch is adjacent to a grid border"""
    return (
        uppermost(patch) == 0
        or leftmost(patch) == 0
        or lowermost(patch) == len(grid) - 1
        or rightmost(patch) == len(grid[0]) - 1
    )


def centerofmass(patch: Patch) -> IntegerTuple:
    """Center of mass"""
    return tuple(map(lambda x: sum(x) // len(patch), zip(*toindices(patch), strict=False)))


def palette(element: Element) -> IntegerSet:
    """Colors occurring in object or grid"""
    if isinstance(element, tuple):
        return frozenset({v for r in element for v in r})
    return frozenset({v for v, _ in element})


def numcolors(element: Element) -> IntegerSet:
    """Number of colors occurring in object or grid"""
    return len(palette(element))


def color(obj: Object) -> Integer:
    """Color of object"""
    return next(iter(obj))[0]


def toobject(patch: Patch, grid: Grid) -> Object:
    """Object from patch and grid"""
    h, w = len(grid), len(grid[0])
    return frozenset((grid[i][j], (i, j)) for i, j in toindices(patch) if 0 <= i < h and 0 <= j < w)


def asobject(grid: Grid) -> Object:
    """Conversion of grid to object"""
    return frozenset((v, (i, j)) for i, r in enumerate(grid) for j, v in enumerate(r))


def rot90(grid: Grid) -> Grid:
    """Quarter clockwise rotation"""
    return tuple(row for row in zip(*grid[::-1], strict=False))


def rot180(grid: Grid) -> Grid:
    """Half rotation"""
    return tuple(tuple(row[::-1]) for row in grid[::-1])


def rot270(grid: Grid) -> Grid:
    """Quarter anticlockwise rotation"""
    return tuple(tuple(row[::-1]) for row in zip(*grid[::-1], strict=False))[::-1]


def hmirror(piece: Piece) -> Piece:
    """Mirroring along horizontal"""
    if isinstance(piece, tuple):
        return piece[::-1]
    d = ulcorner(piece)[0] + lrcorner(piece)[0]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (d - i, j)) for v, (i, j) in piece)
    return frozenset((d - i, j) for i, j in piece)


def vmirror(piece: Piece) -> Piece:
    """Mirroring along vertical"""
    if isinstance(piece, tuple):
        return tuple(row[::-1] for row in piece)
    d = ulcorner(piece)[1] + lrcorner(piece)[1]
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (i, d - j)) for v, (i, j) in piece)
    return frozenset((i, d - j) for i, j in piece)


def dmirror(piece: Piece) -> Piece:
    """Mirroring along diagonal"""
    if isinstance(piece, tuple):
        return tuple(zip(*piece, strict=False))
    a, b = ulcorner(piece)
    if isinstance(next(iter(piece))[1], tuple):
        return frozenset((v, (j - b + a, i - a + b)) for v, (i, j) in piece)
    return frozenset((j - b + a, i - a + b) for i, j in piece)


def cmirror(piece: Piece) -> Piece:
    """Mirroring along counterdiagonal"""
    if isinstance(piece, tuple):
        return tuple(zip(*(r[::-1] for r in piece[::-1]), strict=False))
    return vmirror(dmirror(vmirror(piece)))


def fill(grid: Grid, value: Integer, patch: Patch) -> Grid:
    """Fill value at indices"""
    h, w = len(grid), len(grid[0])
    grid_filled = list(list(row) for row in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            grid_filled[i][j] = value
    return tuple(tuple(row) for row in grid_filled)


def paint(grid: Grid, obj: Object) -> Grid:
    """Paint object to grid"""
    h, w = len(grid), len(grid[0])
    grid_painted = list(list(row) for row in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            grid_painted[i][j] = value
    return tuple(tuple(row) for row in grid_painted)


def underfill(grid: Grid, value: Integer, patch: Patch) -> Grid:
    """Fill value at indices that are background"""
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for i, j in toindices(patch):
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def underpaint(grid: Grid, obj: Object) -> Grid:
    """Paint object to grid where there is background"""
    h, w = len(grid), len(grid[0])
    bg = mostcolor(grid)
    g = list(list(r) for r in grid)
    for value, (i, j) in obj:
        if 0 <= i < h and 0 <= j < w:
            if g[i][j] == bg:
                g[i][j] = value
    return tuple(tuple(r) for r in g)


def hupscale(grid: Grid, factor: Integer) -> Grid:
    """Upscale grid horizontally"""
    g = tuple()
    for row in grid:
        r = tuple()
        for value in row:
            r = r + tuple(value for num in range(factor))
        g = g + (r,)
    return g


def vupscale(grid: Grid, factor: Integer) -> Grid:
    """Upscale grid vertically"""
    g = tuple()
    for row in grid:
        g = g + tuple(row for num in range(factor))
    return g


def upscale(element: Element, factor: Integer) -> Element:
    """Upscale object or grid"""
    if isinstance(element, tuple):
        g = tuple()
        for row in element:
            upscaled_row = tuple()
            for value in row:
                upscaled_row = upscaled_row + tuple(value for num in range(factor))
            g = g + tuple(upscaled_row for num in range(factor))
        return g
    else:
        if len(element) == 0:
            return frozenset()
        di_inv, dj_inv = ulcorner(element)
        di, dj = (-di_inv, -dj_inv)
        normed_obj = shift(element, (di, dj))
        o = set()
        for value, (i, j) in normed_obj:
            for io in range(factor):
                for jo in range(factor):
                    o.add((value, (i * factor + io, j * factor + jo)))
        return shift(frozenset(o), (di_inv, dj_inv))


def downscale(grid: Grid, factor: Integer) -> Grid:
    """Downscale grid"""
    h, w = len(grid), len(grid[0])
    g = tuple()
    for i in range(h):
        r = tuple()
        for j in range(w):
            if j % factor == 0:
                r = r + (grid[i][j],)
        g = g + (r,)
    h = len(g)
    dsg = tuple()
    for i in range(h):
        if i % factor == 0:
            dsg = dsg + (g[i],)
    return dsg


def hconcat(a: Grid, b: Grid) -> Grid:
    """Concatenate two grids horizontally"""
    return tuple(i + j for i, j in zip(a, b, strict=False))


def vconcat(a: Grid, b: Grid) -> Grid:
    """Concatenate two grids vertically"""
    return a + b


def subgrid(patch: Patch, grid: Grid) -> Grid:
    """Smallest subgrid containing object"""
    return crop(grid, ulcorner(patch), shape(patch))


def hsplit(grid: Grid, n: Integer) -> Tuple:
    """Split grid horizontally"""
    h, w = len(grid), len(grid[0]) // n
    offset = len(grid[0]) % n != 0
    return tuple(crop(grid, (0, w * i + i * offset), (h, w)) for i in range(n))


def vsplit(grid: Grid, n: Integer) -> Tuple:
    """Split grid vertically"""
    h, w = len(grid) // n, len(grid[0])
    offset = len(grid) % n != 0
    return tuple(crop(grid, (h * i + i * offset, 0), (h, w)) for i in range(n))


def cellwise(a: Grid, b: Grid, fallback: Integer) -> Grid:
    """Cellwise match of two grids"""
    h, w = len(a), len(a[0])
    resulting_grid = tuple()
    for i in range(h):
        row = tuple()
        for j in range(w):
            a_value = a[i][j]
            value = a_value if a_value == b[i][j] else fallback
            row = row + (value,)
        resulting_grid = resulting_grid + (row,)
    return resulting_grid


def replace(grid: Grid, replacee: Integer, replacer: Integer) -> Grid:
    """Color substitution"""
    return tuple(tuple(replacer if v == replacee else v for v in r) for r in grid)


def switch(grid: Grid, a: Integer, b: Integer) -> Grid:
    """Color switching"""
    return tuple(tuple(v if (v != a and v != b) else {a: b, b: a}[v] for v in r) for r in grid)


def center(patch: Patch) -> IntegerTuple:
    """Center of the patch"""
    return (uppermost(patch) + height(patch) // 2, leftmost(patch) + width(patch) // 2)


def position(a: Patch, b: Patch) -> IntegerTuple:
    """Relative position between two patches"""
    ia, ja = center(toindices(a))
    ib, jb = center(toindices(b))
    if ia == ib:
        return (0, 1 if ja < jb else -1)
    elif ja == jb:
        return (1 if ia < ib else -1, 0)
    elif ia < ib:
        return (1, 1 if ja < jb else -1)
    elif ia > ib:
        return (-1, 1 if ja < jb else -1)


def index(grid: Grid, loc: IntegerTuple) -> Integer:
    """Color at location"""
    i, j = loc
    h, w = len(grid), len(grid[0])
    if not (0 <= i < h and 0 <= j < w):
        return None
    return grid[loc[0]][loc[1]]


def canvas(value: Integer, dimensions: IntegerTuple) -> Grid:
    """Grid construction"""
    return tuple(tuple(value for j in range(dimensions[1])) for i in range(dimensions[0]))


def corners(patch: Patch) -> Indices:
    """Indices of corners"""
    return frozenset({ulcorner(patch), urcorner(patch), llcorner(patch), lrcorner(patch)})


def connect(a: IntegerTuple, b: IntegerTuple) -> Indices:
    """Line between two points"""
    ai, aj = a
    bi, bj = b
    si = min(ai, bi)
    ei = max(ai, bi) + 1
    sj = min(aj, bj)
    ej = max(aj, bj) + 1
    if ai == bi:
        return frozenset((ai, j) for j in range(sj, ej))
    elif aj == bj:
        return frozenset((i, aj) for i in range(si, ei))
    elif bi - ai == bj - aj:
        return frozenset((i, j) for i, j in zip(range(si, ei), range(sj, ej), strict=False))
    elif bi - ai == aj - bj:
        return frozenset(
            (i, j) for i, j in zip(range(si, ei), range(ej - 1, sj - 1, -1), strict=False)
        )
    return frozenset()


def cover(grid: Grid, patch: Patch) -> Grid:
    """Remove object from grid"""
    return fill(grid, mostcolor(grid), toindices(patch))


def trim(grid: Grid) -> Grid:
    """Trim border of grid"""
    return tuple(r[1:-1] for r in grid[1:-1])


def move(grid: Grid, obj: Object, offset: IntegerTuple) -> Grid:
    """Move object on grid"""
    return paint(cover(grid, obj), shift(obj, offset))


def tophalf(grid: Grid) -> Grid:
    """Upper half of grid"""
    return grid[: len(grid) // 2]


def bottomhalf(grid: Grid) -> Grid:
    """Lower half of grid"""
    return grid[len(grid) // 2 + len(grid) % 2 :]


def lefthalf(grid: Grid) -> Grid:
    """Left half of grid"""
    return rot270(tophalf(rot90(grid)))


def righthalf(grid: Grid) -> Grid:
    """Right half of grid"""
    return rot270(bottomhalf(rot90(grid)))


def vfrontier(location: IntegerTuple) -> Indices:
    """Vertical frontier"""
    return frozenset((i, location[1]) for i in range(30))


def hfrontier(location: IntegerTuple) -> Indices:
    """Horizontal frontier"""
    return frozenset((location[0], j) for j in range(30))


def backdrop(patch: Patch) -> Indices:
    """Indices in bounding box of patch"""
    if len(patch) == 0:
        return frozenset({})
    indices = toindices(patch)
    si, sj = ulcorner(indices)
    ei, ej = lrcorner(patch)
    return frozenset((i, j) for i in range(si, ei + 1) for j in range(sj, ej + 1))


def delta(patch: Patch) -> Indices:
    """Indices in bounding box but not part of patch"""
    if len(patch) == 0:
        return frozenset({})
    return backdrop(patch) - toindices(patch)


def gravitate(source: Patch, destination: Patch) -> IntegerTuple:
    """Direction to move source until adjacent to destination"""
    si, sj = center(source)
    di, dj = center(destination)
    i, j = 0, 0
    if vmatching(source, destination):
        i = 1 if si < di else -1
    else:
        j = 1 if sj < dj else -1
    gi, gj = i, j
    c = 0
    while not adjacent(source, destination) and c < 42:
        c += 1
        gi += i
        gj += j
        source = shift(source, (i, j))
    return (gi - i, gj - j)


def inbox(patch: Patch) -> Indices:
    """Inbox for patch"""
    ai, aj = uppermost(patch) + 1, leftmost(patch) + 1
    bi, bj = lowermost(patch) - 1, rightmost(patch) - 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def outbox(patch: Patch) -> Indices:
    """Outbox for patch"""
    ai, aj = uppermost(patch) - 1, leftmost(patch) - 1
    bi, bj = lowermost(patch) + 1, rightmost(patch) + 1
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def box(patch: Patch) -> Indices:
    """Outline of patch"""
    if len(patch) == 0:
        return patch
    ai, aj = ulcorner(patch)
    bi, bj = lrcorner(patch)
    si, sj = min(ai, bi), min(aj, bj)
    ei, ej = max(ai, bi), max(aj, bj)
    vlines = {(i, sj) for i in range(si, ei + 1)} | {(i, ej) for i in range(si, ei + 1)}
    hlines = {(si, j) for j in range(sj, ej + 1)} | {(ei, j) for j in range(sj, ej + 1)}
    return frozenset(vlines | hlines)


def shoot(start: IntegerTuple, direction: IntegerTuple) -> Indices:
    """Line from starting point and direction"""
    return connect(start, (start[0] + 42 * direction[0], start[1] + 42 * direction[1]))


def occurrences(grid: Grid, obj: Object) -> Indices:
    """Locations of occurrences of object in grid"""
    occs = set()
    normed = normalize(obj)
    h, w = len(grid), len(grid[0])
    oh, ow = shape(obj)
    h2, w2 = h - oh + 1, w - ow + 1
    for i in range(h2):
        for j in range(w2):
            occurs = True
            for v, (a, b) in shift(normed, (i, j)):
                if not (0 <= a < h and 0 <= b < w and grid[a][b] == v):
                    occurs = False
                    break
            if occurs:
                occs.add((i, j))
    return frozenset(occs)


def frontiers(grid: Grid) -> Objects:
    """Set of frontiers"""
    h, w = len(grid), len(grid[0])
    row_indices = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    column_indices = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    hfrontiers = frozenset({
        frozenset({(grid[i][j], (i, j)) for j in range(w)}) for i in row_indices
    })
    vfrontiers = frozenset({
        frozenset({(grid[i][j], (i, j)) for i in range(h)}) for j in column_indices
    })
    return hfrontiers | vfrontiers


def compress(grid: Grid) -> Grid:
    """Removes frontiers from grid"""
    ri = tuple(i for i, r in enumerate(grid) if len(set(r)) == 1)
    ci = tuple(j for j, c in enumerate(dmirror(grid)) if len(set(c)) == 1)
    return tuple(
        tuple(v for j, v in enumerate(r) if j not in ci) for i, r in enumerate(grid) if i not in ri
    )


def hperiod(obj: Object) -> Integer:
    """Horizontal periodicity"""
    normalized = normalize(obj)
    w = width(normalized)
    for p in range(1, w):
        offsetted = shift(normalized, (0, -p))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if j >= 0})
        if pruned.issubset(normalized):
            return p
    return w


def vperiod(obj: Object) -> Integer:
    """Vertical periodicity"""
    normalized = normalize(obj)
    h = height(normalized)
    for p in range(1, h):
        offsetted = shift(normalized, (-p, 0))
        pruned = frozenset({(c, (i, j)) for c, (i, j) in offsetted if i >= 0})
        if pruned.issubset(normalized):
            return p
    return h
