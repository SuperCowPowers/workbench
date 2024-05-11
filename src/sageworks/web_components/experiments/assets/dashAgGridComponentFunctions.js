var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

dagcomponentfuncs.Link = function (props) {
    return React.createElement(
        'a',
        {href:  props.value},
        props.value
    );
};