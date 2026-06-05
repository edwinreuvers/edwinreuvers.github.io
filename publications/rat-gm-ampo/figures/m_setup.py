#| label: fig-m-setup
#| fig-cap: Representation of the experimental setup that provided full control of MTC length
#|   and stimulation while measuring *m. gastrocnemius medialis* (GM) force. GM was carefully
#|   exposed from its surrounding tissue and positioned in the setup such that GM pulled in its natural direction,
#|   while the femur and foot were securely fixated. The distal end of the calcaneal tendon was connected to a
#|   motor via a steel rod. The distal tendon of *m. gastrocnemius lateralis* and *m. plantaris* was connected to a
#|   second motor. A cuff-electrode was placed on *n. ischiadicus*. *N. peroneus*, *n. suralis* and the branch of
#|   *n. ischiadicus* innervating *m. gastrocnemius lateralis* and *m. soleus* were cut such that only GM was
#|   innervated.

# %%
from PIL import Image
import matplotlib.pyplot as plt

# %%
# Load PNG
img = Image.open("m_setup.png")

# Display
plt.imshow(img)
plt.axis("off")
plt.show()