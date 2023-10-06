Quickstart
==========

Initializing a task::

.. code-block:: python

    from ODRS.ODRS.api.ODRS import ODRS
    #init object with parameters
    odrs = ODRS(job="object_detection", data_path = 'full_data_path', classes = "classes.txt",
                    img_size = "512", batch_size = "25", epochs = "300",
                    model = 'yolov8x6', gpu_count = 1, select_gpu = "0", config_path = "dataset.yaml",
                    split_train_value = 0.6, split_test_value = 0.35, split_val_value = 0.05)
Starting training:

.. code-block:: python

    from ODRS.ODRS.api.ODRS import ODRS
    odrs.fit()