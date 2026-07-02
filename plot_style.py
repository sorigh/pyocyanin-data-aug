"""
Shared Plotly styling for every figure in this project.

Single source of truth for `apply_default_plotly_layout` so that
any notebook renders figures with an identical look. For import.
"""

PLOTLY_FIGURE_HEIGHT_PX = 520
PLOTLY_FIGURE_WIDTH_PX = 880
PLOTLY_TEMPLATE = 'plotly_white'


def apply_default_plotly_layout(figure, title_text=None,
                                 xaxis_title='Potential E (V)',
                                 yaxis_title='Current I (µA)'):
    figure.update_layout(
        height=PLOTLY_FIGURE_HEIGHT_PX, width=PLOTLY_FIGURE_WIDTH_PX,
        template=PLOTLY_TEMPLATE,
        title=dict(text=title_text, x=0.5, xanchor='center') if title_text else None,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    figure.update_xaxes(showgrid=True, minor=dict(showgrid=True))
    figure.update_yaxes(showgrid=True, minor=dict(showgrid=True))
    return figure
