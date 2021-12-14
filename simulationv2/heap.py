class Heap:
    '''
    Es un minheap
    '''

    def __init__(self, key, max_size=4):
        self.size = 0
        self.heap = [0] * max_size
        self.max_size = max_size
        self.key = key

    def swap(self, i, j):
        temp = self.heap[i]
        self.heap[i] = self.heap[j]
        self.heap[j] = temp

    def parent(self, i):
        j = (i - 1) // 2
        return j

    def left(self, i):
        j = 2 * i + 1
        return j

    def right(self, i):
        j = 2 * i + 2
        return j

    def siftup(self, i):
        p = self.parent(i)
        while i != 0 and self.key(self.heap[i]) < self.key(self.heap[p]):
            self.swap(self, i, p)
            i = p

    def siftdown(self, i):
        l = self.left(i)
        r = self.right(i)
        s = i
        if l < self.size and self.key(self.heap[l]) < self.key(self.heap[i]):
            s = l
        if r < self.size and self.key(self.heap[r]) < self.key(self.heap[s]):
            s = r
        if s != i:
            self.swap(i, s)
            self.siftdown(s)

    def push(self, elem):
        if self.size != self.max_size:
            i = self.size
            self.heap[i] = elem
            self.size += 1
            self.siftup(i)
        else:
            if self.key(self.heap[0]) < self.key(elem):
                self.heap[0] = elem
                self.siftdown(0)

    def pop(self):
        root = self.heap[0]
        self.size -= 1
        self.heap[0] = self.heap[self.size]
        self.siftdown(0)
        return root
