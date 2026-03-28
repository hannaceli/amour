/**
 * Google Apps Script to auto-create the Travel & Logistics Form
 * for the 10-Year Anniversary Celebration (July 4-11, 2026).
 *
 * HOW TO USE:
 * 1. Go to https://script.google.com
 * 2. Create a new project
 * 3. Paste this entire script into the editor
 * 4. Click the Run button (▶) to execute createTravelForm()
 * 5. Grant the required permissions when prompted
 * 6. The form URL will appear in the execution log (View > Logs)
 * 7. Copy that URL and update the website's Travel Form link
 */

function createTravelForm() {
  var form = FormApp.create('Travel & Logistics — Anniversary Week (July 4-11, 2026)');

  form.setDescription(
    'Help us coordinate transport and logistics for the week in Burgundy! ' +
    'Please fill this out by mid-June 2026 so we can organise transfers and meals.\n\n' +
    'You can edit your response later if plans change.'
  );

  form.setConfirmationMessage(
    'Thank you! Your response has been recorded. ' +
    'If your plans change, you can edit your response using the link in your confirmation email.'
  );

  form.setAllowResponseEdits(true);
  form.setCollectEmail(true);

  // --- Section 1: Who ---
  form.addSectionHeaderItem()
    .setTitle('About You')
    .setHelpText('Tell us who is coming.');

  form.addTextItem()
    .setTitle('Full name')
    .setRequired(true);

  form.addTextItem()
    .setTitle('Number of adults in your party')
    .setRequired(true);

  form.addTextItem()
    .setTitle('Number of children in your party')
    .setHelpText('Enter 0 if no children.')
    .setRequired(true);

  form.addTextItem()
    .setTitle('Ages of children (if any)')
    .setHelpText('e.g. "5, 8" — leave blank if no children.');

  // --- Section 2: Arrival ---
  form.addSectionHeaderItem()
    .setTitle('Arrival')
    .setHelpText('When and how are you arriving?');

  form.addDateItem()
    .setTitle('Arrival date')
    .setRequired(true);

  form.addTextItem()
    .setTitle('Approximate arrival time')
    .setHelpText('e.g. "Around 3pm" or "Evening"')
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('How are you arriving?')
    .setChoiceValues(['By car', 'By train', 'By plane + train', 'By plane + rental car', 'Other'])
    .setRequired(true);

  form.addTextItem()
    .setTitle('If train/plane: arrival station or airport')
    .setHelpText('e.g. "Le Creusot TGV", "Lyon Saint-Exupéry", "Paris CDG"');

  form.addTextItem()
    .setTitle('If train/plane: flight or train number (if known)')
    .setHelpText('e.g. "TGV 6721" or "AF 1234"');

  form.addMultipleChoiceItem()
    .setTitle('Do you need a transfer from station/airport to the venue on arrival?')
    .setChoiceValues(['Yes please!', 'No, I have my own transport', 'Maybe — not sure yet'])
    .setRequired(true);

  // --- Section 3: Departure ---
  form.addSectionHeaderItem()
    .setTitle('Departure')
    .setHelpText('When and how are you leaving?');

  form.addDateItem()
    .setTitle('Departure date')
    .setRequired(true);

  form.addTextItem()
    .setTitle('Approximate departure time')
    .setHelpText('e.g. "Morning", "After lunch", "Around 2pm"')
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('How are you departing?')
    .setChoiceValues(['By car', 'By train', 'By plane (via train to airport)', 'By plane (via rental car)', 'Other'])
    .setRequired(true);

  form.addTextItem()
    .setTitle('If train/plane: departure station or airport')
    .setHelpText('e.g. "Le Creusot TGV", "Lyon Saint-Exupéry"');

  form.addTextItem()
    .setTitle('If train/plane: flight or train number (if known)');

  form.addMultipleChoiceItem()
    .setTitle('Do you need a transfer from the venue to station/airport on departure?')
    .setChoiceValues(['Yes please!', 'No, I have my own transport', 'Maybe — not sure yet'])
    .setRequired(true);

  // --- Section 4: Car sharing ---
  form.addSectionHeaderItem()
    .setTitle('Car Sharing')
    .setHelpText('If you\'re driving, can you help others get to/from the venue?');

  form.addMultipleChoiceItem()
    .setTitle('If you are driving: can you offer spare seats to others?')
    .setChoiceValues(['Yes, happy to help!', 'Maybe — depends on timing', 'No, my car will be full', 'I\'m not driving']);

  form.addTextItem()
    .setTitle('If yes/maybe: how many spare seats can you offer?')
    .setHelpText('e.g. "2 seats" or "3 seats but only from Le Creusot"');

  // --- Section 5: Food ---
  form.addSectionHeaderItem()
    .setTitle('Food & Dietary Needs')
    .setHelpText('Help us plan meals for the week.');

  form.addCheckboxItem()
    .setTitle('Any dietary restrictions or allergies?')
    .setChoiceValues([
      'None',
      'Vegetarian',
      'Vegan',
      'Gluten-free',
      'Dairy-free',
      'Nut allergy',
      'Shellfish allergy',
      'Other (please specify below)'
    ]);

  form.addTextItem()
    .setTitle('If "Other", please specify dietary needs')
    .setHelpText('Also note any children\'s specific dietary needs here.');

  // --- Section 6: Notes ---
  form.addSectionHeaderItem()
    .setTitle('Anything Else?');

  form.addParagraphTextItem()
    .setTitle('Any other notes, questions, or requests?')
    .setHelpText('e.g. "We\'ll arrive a day early and stay in Autun", "Need a crib for the baby", etc.');

  // Log the form URL
  Logger.log('Form created successfully!');
  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Published URL: ' + form.getPublishedUrl());
  Logger.log('');
  Logger.log('NEXT STEPS:');
  Logger.log('1. Open the Edit URL above to customize colors/theme');
  Logger.log('2. Copy the Published URL');
  Logger.log('3. Update the website link in src/components/TravelForm.astro');
  Logger.log('   Replace YOUR_FORM_ID with the actual form ID from the URL');
}
