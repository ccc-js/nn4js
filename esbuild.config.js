const esbuild = require('esbuild');
const path = require('path');

esbuild.build({
  entryPoints: ['src/index.js'],
  bundle: true,
  outfile: 'web/nn4js.js',
  format: 'iife',
  globalName: 'nn4js',
  minify: true,
  target: ['es2020'],
  logLevel: 'info'
}).catch(() => process.exit(1));