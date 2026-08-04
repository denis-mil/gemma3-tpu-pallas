/* quiz.js — reusable retrieval-practice widget.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   Quiz.mount('#q', {
 *     title: 'Retrieval practice',
 *     questions: [{
 *       stem: 'How many live tiles?',          // may contain HTML
 *       options: ['24 tiles', '26 tiles'],     // keep these EQUAL length
 *       answer: 1,                             // index into options
 *       why: 'Because ...'                     // shown after answering
 *     }]
 *   });
 *
 * Design notes:
 *  - Immediate feedback: the loop is tightest when it closes on click.
 *  - One attempt per question. Retrieval only builds storage strength if the
 *    attempt is committed; letting the user reshuffle turns it into recognition.
 *  - Options are NOT shuffled. They are authored in a deliberate order (usually
 *    numeric) so that position carries no signal about correctness.
 *  - In dev, warns when option lengths differ enough to leak the answer.
 */
(function (global) {
  'use strict';

  var LENGTH_TOLERANCE = 3; // characters

  function warnIfLengthLeaks(options, index) {
    var lens = options.map(function (o) { return o.replace(/<[^>]*>/g, '').length; });
    var min = Math.min.apply(null, lens);
    var max = Math.max.apply(null, lens);
    if (max - min > LENGTH_TOLERANCE) {
      console.warn(
        '[quiz] Q' + (index + 1) + ': option lengths ' + lens.join('/') +
        ' vary by ' + (max - min) + ' chars — formatting may leak the answer.'
      );
    }
    var words = options.map(function (o) { return o.trim().split(/\s+/).length; });
    if (new Set(words).size > 1) {
      console.warn('[quiz] Q' + (index + 1) + ': option word counts ' + words.join('/') + ' differ.');
    }
  }

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }

    var questions = spec.questions || [];
    var answered = 0;
    var correct = 0;

    root.classList.add('widget');
    root.innerHTML = '';

    if (spec.title) {
      var t = document.createElement('p');
      t.className = 'widget-title';
      t.textContent = spec.title;
      root.appendChild(t);
    }

    var score = document.createElement('p');
    score.className = 'q-score';

    questions.forEach(function (q, qi) {
      warnIfLengthLeaks(q.options, qi);

      var block = document.createElement('div');
      block.className = 'q';

      var stem = document.createElement('p');
      stem.className = 'q-stem';
      stem.innerHTML = '<b>' + (qi + 1) + '.</b> ' + q.stem;
      block.appendChild(stem);

      var opts = document.createElement('div');
      opts.className = 'q-opts';

      var feedback = document.createElement('div');
      feedback.className = 'q-feedback';
      feedback.hidden = true;

      var buttons = q.options.map(function (text, oi) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'q-opt';
        b.innerHTML = text;
        b.addEventListener('click', function () {
          if (block.dataset.done) { return; }
          block.dataset.done = '1';

          buttons.forEach(function (other, otherIndex) {
            other.disabled = true;
            if (otherIndex === q.answer) { other.classList.add('correct'); }
            else if (otherIndex === oi) { other.classList.add('wrong'); }
          });

          var got = oi === q.answer;
          answered += 1;
          if (got) { correct += 1; }

          feedback.innerHTML =
            '<b>' + (got ? 'Correct.' : 'Not quite.') + '</b> ' + q.why;
          feedback.hidden = false;

          score.textContent =
            answered + ' of ' + questions.length + ' answered · ' +
            correct + ' correct' +
            (answered === questions.length && correct === questions.length
              ? ' — you can now read the mask as a work map.'
              : '');
        });
        opts.appendChild(b);
        return b;
      });

      block.appendChild(opts);
      block.appendChild(feedback);
      root.appendChild(block);
    });

    root.appendChild(score);
  }

  global.Quiz = { mount: mount };
})(window);
