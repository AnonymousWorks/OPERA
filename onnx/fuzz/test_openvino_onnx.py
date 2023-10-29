import onnx
from onnx import helper
import numpy as np
import onnxruntime

import ipywidgets as widgets
import openvino as ov

import os
import traceback
import re
import random
import string


import logging
logging.basicConfig(level=logging.ERROR)

np.random.seed(2023)

def convert_kwargs(kwargs):
    new_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and value.startswith('['):
            try:
                value_np = np.fromstring(value.strip('[]'), sep=' ', dtype=np.float32)
                new_kwargs[key] = onnx.numpy_helper.from_array(value_np)
            except Exception as e:
                new_kwargs[key] = value
        else:
            new_kwargs[key] = value
    return new_kwargs

def extract_crash_message(e):
    tb = traceback.extract_tb(e.__traceback__)
    file_name, line_number, _, _ = tb[-1]
    # file_name = file_name[len("/workplace/software/dlc/dlc_/"):]
    exc_type = type(e).__name__
    stack_trace = str(e).strip().split("\n")[-1]
    if stack_trace.endswith(':'):
        stack_trace = stack_trace[:-1]
    stack_trace = stack_trace.split(':')[-1].strip()
    pattern = r"[\[\(].*?[\]\)]"
    stack_trace = re.sub(pattern, "", stack_trace)
    print(f">>>>>>>>>>>>>>>>>>>Bug Info: {stack_trace}")
    # exc_value = str(e)
    # exception_info = traceback.format_exception_only(exc_type, exc_value)
    # crash_message = f"{exception_info}_{file_name}_{line_number}"
    # onnx.helper.get_attribute_value()
    crash_message = f"{exc_type}_{file_name}_{line_number}_{stack_trace}"
    return crash_message


def record_bug(bug_id, bug_type, op, crash_message=''):
    bug_info_str = f"{bug_id}\t{bug_type}\t{op}\t{crash_message}\n"

    with open("detected_bugs_new.txt", 'a', encoding='utf-8') as f:
        f.write(bug_info_str)

onnx_dtype_mapping = {
    'FLOAT': onnx.TensorProto.FLOAT,
    'FLOAT16': onnx.TensorProto.FLOAT16,
    'DOUBLE': onnx.TensorProto.DOUBLE,
    'INT8': onnx.TensorProto.INT8,
    'INT16': onnx.TensorProto.INT16,
    'INT32': onnx.TensorProto.INT32,
    'INT64': onnx.TensorProto.INT64,
    'UINT8': onnx.TensorProto.UINT8,
    'UINT16': onnx.TensorProto.UINT16,
    'UINT32': onnx.TensorProto.UINT32,
    'UINT64': onnx.TensorProto.UINT64,
    'BOOL': onnx.TensorProto.BOOL,
    'STRING': onnx.TensorProto.STRING
}

dlc_dtype_mapping = {
    'FLOAT': np.float32,
    'FLOAT16': np.float16,
    'DOUBLE': np.float64,
    'INT8': np.int8,
    'INT16': np.int16,
    'INT32': np.int32,
    'INT64': np.int64,
    'UINT8': np.uint8,
    'UINT16': np.uint16,
    'UINT32': np.uint32,
    'UINT64': np.uint64,
    'BOOL': np.bool_,
    'STRING': np.str_
}

def assign_input_data(shape, dtype):    
    # assign value to input_data
    if dtype == onnx.TensorProto.FLOAT:
        return np.random.rand(*shape).astype(np.float32)
    elif dtype == onnx.TensorProto.FLOAT16:
        return np.random.rand(*shape).astype(np.float16)
    elif dtype == onnx.TensorProto.DOUBLE:
        return np.random.rand(*shape).astype(np.float64)
    elif dtype == onnx.TensorProto.INT64:
        return np.random.randint(0, 2, shape, dtype=np.int64)
    elif dtype == onnx.TensorProto.INT32:
        return np.random.randint(0, 2, shape, dtype=np.int32)
    elif dtype == onnx.TensorProto.INT16:
        return np.random.randint(0, 2, shape, dtype=np.int16)
    elif dtype == onnx.TensorProto.INT8:
        return np.random.randint(0, 2, shape, dtype=np.int8)
    elif dtype == onnx.TensorProto.UINT8:
        return np.random.randint(-2, 2, shape, dtype=np.uint8)
    elif dtype == onnx.TensorProto.UINT16:
        return np.random.randint(-2, 2, shape, dtype=np.uint16)
    elif dtype == onnx.TensorProto.UINT32:
        return np.random.randint(-2, 2, shape, dtype=np.uint32)
    elif dtype == onnx.TensorProto.UINT64:
        return np.random.randint(-2, 2, shape, dtype=np.uint64)
    elif dtype == onnx.TensorProto.BOOL:
        return np.random.choice([True, False], shape)
    elif dtype == onnx.TensorProto.STRING:
        return np.array([''.join(random.choices(string.ascii_letters + string.digits, k=10)) for _ in range(np.prod(shape))]).reshape(shape)

def make_sub_graph(string):
    op_type, kwargs, input_name, input_shape, input_dtype, output_name, output_shape, output_dtype = \
    string.split("op_type=")[1].split(", kwargs=")[0],\
    string.split(", kwargs=")[1].split(", input_name=")[0],\
    string.split(", input_name=")[1].split(", input_shape=")[0],\
    string.split(", input_shape=")[1].split(", input_dtype=")[0],\
    string.split(", input_dtype=")[1].split(", output_name=")[0],\
    string.split(", output_name=")[1].split(", output_shape=")[0],\
    string.split(", output_shape=")[1].split(", output_dtype=")[0],\
    string.split(", output_dtype=")[1]
    op_type = eval(op_type)
    kwargs_dict = eval(kwargs)
    input_name = eval(input_name)
    input_shape = eval(input_shape)
    input_dtype = eval(input_dtype)
    output_name = eval(output_name)
    output_shape = eval(output_shape)
    output_dtype = eval(output_dtype)
    kwargs = convert_kwargs(kwargs_dict)
    node_def = helper.make_node(
        op_type,  # node name
        input_name,  # inputs
        output_name,  # outputs
        **kwargs
    )
    # if shape is [], change it to [1]
    new_input_shape = []
    for shape in input_shape:
        if not shape:
            new_input_shape.append([1])
        else:
            new_input_shape.append(shape)
    new_input_shape = tuple(new_input_shape)
    input_shape = new_input_shape

    input_dtype_onnx = [onnx_dtype_mapping[dtype] for dtype in input_dtype]
    output_dtype_onnx = [onnx_dtype_mapping[dtype] for dtype in output_dtype]
    graph_def = helper.make_graph(
        [node_def],
        op_type,
        inputs=[
            helper.make_tensor_value_info(name, dtype, shape)
            for name, shape, dtype in zip(input_name, input_shape, input_dtype_onnx)
        ],
        outputs=[
            helper.make_tensor_value_info(name, dtype, shape)
            for name, shape, dtype in zip(output_name, output_shape, output_dtype_onnx)
        ]
    )
    return graph_def

def make_graph(op_type, kwargs, input_name, input_shape, input_dtype, output_name, output_shape, output_dtype, count=0, **unused_kwargs):
    print(count)
    try:
        kwargs = convert_kwargs(kwargs)
        # Create ONNX graph
        if op_type == 'If':
            else_branch = kwargs['else_branch'][len("make_graph("):-1]
            then_branch = kwargs['then_branch'][len("make_graph("):-1]
            else_graph = make_sub_graph(else_branch)
            then_graph = make_sub_graph(then_branch)
            node_def = helper.make_node(
                op_type, # node name
                input_name, # inputs
                output_name, # outputs
                else_branch=else_graph,
                then_branch=then_graph
            )
        elif op_type == 'Scan' or op_type == 'Loop':
            body = kwargs['body'][len("make_graph("):-1]
            body_graph = make_sub_graph(body)
            node_def = helper.make_node(
                op_type,
                input_name,
                output_name,
                body=body_graph
            )
        else:
            node_def = helper.make_node(
                op_type, # node name
                input_name, # inputs
                output_name, # outputs
                **kwargs
            )
        # if shape is [], change it to [1]
        new_input_shape = []
        for shape in input_shape:
            if not shape:
                new_input_shape.append([1])
            else:
                new_input_shape.append(shape)
        new_input_shape = tuple(new_input_shape)
        input_shape = new_input_shape

        input_dtype_onnx = [onnx_dtype_mapping[dtype] for dtype in input_dtype]
        output_dtype_onnx = [onnx_dtype_mapping[dtype] for dtype in output_dtype]
        graph_def = helper.make_graph(
            [node_def],
            op_type,
            inputs=[
                helper.make_tensor_value_info(name, dtype, shape)
                for name, shape, dtype in zip(input_name, input_shape, input_dtype_onnx)
            ],
            outputs=[
                helper.make_tensor_value_info(name, dtype, shape)
                for name, shape, dtype in zip(output_name, output_shape, output_dtype_onnx)
            ]
        )
        if op_type.startswith("Sequence"):
            graph_def = helper.make_graph(
                [node_def],
                op_type,
                inputs=[
                    helper.make_tensor_value_info(name, dtype, shape)
                    for name, shape, dtype in zip(input_name, input_shape, input_dtype_onnx)
                ],
                outputs=[
                    helper.make_tensor_sequence_value_info(name, dtype, shape)
                    for name, shape, dtype in zip(output_name, output_shape, output_dtype_onnx)
                ]
            )
        # Check to see if onnxruntime version is less than 1.15, if so ir_version should
        # be 8 for now. See: https://github.com/microsoft/onnxruntime/issues/15874
        make_model_kwargs = {}
        if onnxruntime.__version__ < "1.15":
            make_model_kwargs = {'ir_version': 8}

        #onnx_model = helper.make_model(onnx_graph, **make_model_kwargs)
        model_def = helper.make_model(graph_def, opset_imports=[onnx.helper.make_opsetid("", 18)], **make_model_kwargs)
        #onnx.save(model_def, 'ReduceL2.onnx')
        # Generate input data
        special_list = ['ConstantOfShape']
        input_data = {}
        for name, shape, dtype in zip(input_name, input_shape, input_dtype_onnx):
            input_data[name] = assign_input_data(shape, dtype)
            if op_type == 'ConstantOfShape':
                input_data[name] = output_shape[0]
            elif op_type == 'Split':
                if name == 'split':
                    split_value = np.squeeze(np.concatenate(output_shape))
                    input_data['split'] = split_value
        # Run the model using ONNX runtime
        sess = onnxruntime.InferenceSession(model_def.SerializeToString())
        onnx_output = sess.run(None, input_data)

        # Load the ONNX model using dlc
        onnx_model = onnx.load_model_from_string(model_def.SerializeToString())
    except Exception as e:
        print("[onnx error]", e)
        return
    dlc_output = compile_onnx(count, onnx_model, input_shape, input_data)
    try:
        input_dtype_dlc = [dlc_dtype_mapping[dtype] for dtype in input_dtype]
        dlc_output = compile_onnx(count, onnx_model, input_shape, input_data)
    except Exception as e:
        if 'support' in str(e) or 'not allowed' in str(e) or "No conversion rule" in str(e):
            print("[Warning] trigger an unsupported behavior")
        else:
            print(f'[Bug in DLC] using test: {op_type}; id= {count}')
            print(e)
            crash_message = extract_crash_message(e)
            record_bug(count, 'crash', op_type, crash_message=crash_message)
        return
    try:
        if len(output_name) > 1:  # multiple output:
            for i in range(len(output_name)):
                np.testing.assert_allclose(onnx_output[i], dlc_output[i], atol=1e-3, rtol=1e-3)
        else:
            np.testing.assert_allclose(onnx_output[0], dlc_output, atol=1e-3, rtol=1e-3)
    except AssertionError as e:
        print(f'[Bug in DLC] using test: {op_type}; id= {count}')
        print(e)
        crash_message = extract_crash_message(e)
        record_bug(count, 'wrong results', op_type, crash_message=crash_message)
    else:
        print("[success] This test case passed!")


def compile_onnx(cnt, model, input_shapes, input_data):
    # [reference](https://docs.openvino.ai/2023.1/openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX.html)
    temp_model_dir = "_temp_model"
    if not os.path.exists(temp_model_dir):
        os.mkdir(temp_model_dir)
    model_path = os.path.join(temp_model_dir, f"{cnt}.onnx")
    onnx.save_model(model, model_path)
    print(input_shapes)
    ov_model = ov.convert_model(model_path)  # input=input_shapes
    ir_path = f"{temp_model_dir}/_temp_OVIR_{cnt}.xml"  # file must ends with 'xml'
    ov.save_model(ov_model, ir_path, compress_to_fp16=False)
    core = ov.Core()
    model = core.read_model(ir_path)

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )

    compiled_model = core.compile_model(model=model, device_name=device.value)

    # show the model structure
    # input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    # network_input_shape = input_key.shape

    result = compiled_model(input_data)[output_key]
    return result


if __name__ == '__main__':
    make_graph(op_type='Constant', kwargs={'value': '0.42'}, input_name=('x',), input_shape=([],), input_dtype=('INT32',), output_name=('y',), output_shape=([10],), output_dtype=('FLOAT',))

