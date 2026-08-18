Getting Started
===============

Installation
------------

Install the package from the repository root in editable mode while developing:

.. code-block:: bash

   pip install -e .

Create And Run An RNN
---------------------

RNNToolkit analyzes standard PyTorch recurrent modules. Configure the module
with ``batch_first=True`` so trajectories use ``[batch, time, features]``.

.. code-block:: python

   import torch

   rnn = torch.nn.RNN(
       input_size=2,
       hidden_size=4,
       nonlinearity="tanh",
       batch_first=True,
   )
   inputs = torch.randn(8, 30, 2)
   h0 = torch.zeros(1, 8, 4)
   states, final_state = rnn(inputs, h0)

Run An Analysis
---------------

The top-level package exports the main analysis classes. For example, compute
local Jacobians around one input and hidden state:

.. code-block:: python

   from rnntoolkit import Linearization

   linearization = Linearization(rnn)
   recurrent_jacobian, input_jacobian = linearization.jacobian(
       inputs[0, 0], states[0, 0]
   )

Next Steps
----------

See :doc:`user_guide` for conceptual workflows and :doc:`examples` for
copy-pastable fixed-point, flow-field, and visualization snippets.
