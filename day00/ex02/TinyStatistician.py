import math

class  TinyStatistician():

    # computes the mean of a given non-empty array x, 
    # using a for-loop and returns the mean as a float, 
    # otherwise None if x is an empty array. 
    # This method should not raise any Exception.
    def mean(self, x):
        array = x[:]
        length = len(array)
        if length == 0:
            return None
        value = 0
        for i in array:
            value += i
        return value/length


    # computes the median, also called the 50th percentile, 
    # of a given non-empty darray x, 
    # using a for-loop and returns the median as a float, 
    # otherwise None if x is an empty array. 
    # This method should not raise any Exception.
    def median(self, x):
        array = x[:]
        length = len(array)
        if length == 0:
            return None

        array.sort()
        pivot = length * (1/2)
        if pivot % 1 == 0:
            return (array[pivot-1] + array[pivot])/2
        else :
            return array[int(pivot)]


    # computes the 1st and 3rd quartiles,
    # also called the 25th percentile and the 75th percentile, 
    # of a given non-empty array x, using a for-loop and returns the quartile as a float, 
    # otherwise None if x is an empty array. 
    # The first parameter is the array and the second parameter is the expected percentile. 
    # This method should not raise any Exception.
    def quartiles(self, x, percentile):
        array = x[:]
        length = len(array)
        if length == 0:
            return None

        array.sort()
        pivot = length * (percentile/100)
        if pivot % 1 == 0:
            return (array[pivot-1] + array[pivot])/2
        else :
            return array[int(pivot)]

    # computes the variance of a given non-empty array x, 
    # using a for-loop and returns the variance as a float, 
    # otherwise None if x is an empty array. 
    # This method should not raise any Exception.
    def var(self, x):
        array = x[:]
        length = len(array)
        if length == 0:
            return None
        mean = self.mean(array)

        value = 0
        for i in array:
            value += (i-mean)*(i-mean)
        return value / length


    # computes the standard deviation of a given non-empty array x, 
    # using a for-loop and returns the standard deviation as a float, 
    # otherwise None if x is an empty array. 
    # This method should not raise any Exception.
    def std(self, x):
        array = x[:]
        length = len(array)
        if length == 0:
            return None

        mean = self.mean(array)

        value = 0
        for i in array:
            value += (i-mean)*(i-mean)

        return math.sqrt(value / length)
