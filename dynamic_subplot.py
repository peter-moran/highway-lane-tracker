from matplotlib import pyplot as plt


class DynamicSubplot:
    def __init__(self, m, n):
        """
        Class for making it easy to 'append' to a subplot, rather than having to handle subplot indices.

        :param m: Number of rows in subplot.
        :param n: Number of columns in subplot.
        """
        self.figure, self.plots = plt.subplots(m, n)
        self.plots = self.plots.flatten()
        self.__curr_plot = -1

    def imshow(self, img, title, cmap=None):
        """Advances to the next plot and displays the image."""
        self.next_subplot()
        self.plots[self.__curr_plot].imshow(img, cmap=cmap)
        self.plots[self.__curr_plot].set_title(title)

    def skip_plot(self):
        """Advances to the next plot and sets it to empty."""
        self.next_subplot()
        self.plots[self.__curr_plot].axis('off')

    def call(self, func_name, *args, **kwargs):
        """Advances to the next plot and calls `func_name(*args, **kwargs)`."""
        self.next_subplot()
        func = getattr(self.plots[self.__curr_plot], func_name)
        func(*args, **kwargs)

    def modify_plot(self, func_name, *args, **kwargs):
        """Calls `func_name(*args, **kwargs)` on the current plot."""
        if self.__curr_plot == -1:
            raise IndexError("There is no plot to modify.")
        func = getattr(self.plots[self.__curr_plot], func_name)
        func(*args, **kwargs)

    def next_subplot(self, n=1):
        """Increments to the next plot."""
        self.__curr_plot += n
        if self.__curr_plot > len(self.plots):
            raise IndexError("You've gone too far forward. There are no more subplots.")

    def last_subplot(self, n=1):
        """Decrements to the next plot."""
        self.__curr_plot -= n
        if self.__curr_plot < 0:
            raise IndexError("You've gone too far back. There are no more subplots.")
