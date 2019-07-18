fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'rx')

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return ln,

def animate(points):
    ln.set_data(0.5, points[0][0])
    return ln,
ani = FuncAnimation(fig, animate(embeddings.eval()),
                    init_func=init, interval=2000, blit=False)
plt.show(block=False)