	u�V��@u�V��@!u�V��@	�C����?�C����?!�C����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$u�V��@K�=�U�?A�!�uq{�@Y��B�i��?*	fffff��@2P
Iterator::Model::Prefetch�lV}���?!o:�ڃX@)�lV}���?1o:�ڃX@:Preprocessing2F
Iterator::Model���H�?!      Y@)�V-�?19d�[	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�C����?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	K�=�U�?K�=�U�?!K�=�U�?      ��!       "      ��!       *      ��!       2	�!�uq{�@�!�uq{�@!�!�uq{�@:      ��!       B      ��!       J	��B�i��?��B�i��?!��B�i��?R      ��!       Z	��B�i��?��B�i��?!��B�i��?JCPU_ONLYY�C����?b 