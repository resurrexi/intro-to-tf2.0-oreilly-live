#!/usr/bin/env python3

from IPython.display import Javascript

def tensorboard_cmd(logs):
    return Javascript("""
        var tb_url = Jupyter.notebook.base_url + "proxy/6006/";
        element.html(
            "Run at the command line: <tt>tensorboard --logdir={log}</tt><br />" +
            "Then open <a href='" + tb_url + "' target='_blank'>" + tb_url + "</a>"
        );
    """.format(log=logs))
