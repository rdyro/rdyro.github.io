const fs = require('fs');
var mjAPI = require("mathjax-node");

mjAPI.config({
  MathJax: {
    // traditional MathJax configuration
  }
});

mjAPI.start();

const input_file = process.argv[2];
const output_file_no_ext = process.argv[3];

fs.readFile(input_file, 'utf8', (err, yourMath) => {
  if (err) {
    console.error(`Error reading the file: ${err}`);
    process.exit(1);
  }

  // Typeset the LaTeX code
  mjAPI.typeset({
    math: yourMath,
    format: "inline-TeX",
    svg: true,
    mml: true,
  }, function (data) {
    if (!data.errors) {

      fs.writeFile(output_file_no_ext + ".svg", data.svg, function (err) {
        if (err) {
          return console.error(err);
        }
        console.log("SVG has been written to:");
        console.log(output_file_no_ext + ".svg");
      });

      fs.writeFile(output_file_no_ext + ".mml", data.mml, function (err) {
        if (err) {
          return console.error(err);
        }
        console.log("MML has been written to:");
        console.log(output_file_no_ext + ".mml");
      });
    }
  });
});